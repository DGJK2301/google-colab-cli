param(
    [string]$Session = "operational-recovery",
    [string]$OrphanEndpoint = "",
    [string]$ColabCommand = "colab"
)

$ErrorActionPreference = "Stop"
$runId = [Guid]::NewGuid().ToString("N").Substring(0, 8)
$sessionName = "$Session-$runId"
$jobId = "monitor-$runId"
$outputRoot = Join-Path ([IO.Path]::GetTempPath()) "colab-monitor-$runId"
$trackedLocally = $false
$primaryError = $null
$cleanupErrors = [System.Collections.Generic.List[string]]::new()

function Invoke-Colab {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    & $ColabCommand @Arguments
}

function Invoke-ColabJson {
    param([string[]]$Arguments)
    $stdout = (Invoke-Colab @Arguments | Out-String)
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "colab $($Arguments -join ' ') failed with $code"
    }
    return $stdout | ConvertFrom-Json -Depth 64
}

try {
    $doctor = Invoke-ColabJson @("doctor", "--json")
    if ($doctor.schema -ne "colab.doctor.v1") {
        throw "Unexpected doctor schema"
    }

    if ($OrphanEndpoint) {
        $attached = Invoke-ColabJson @(
            "attach",
            "--endpoint", $OrphanEndpoint,
            "--session", $sessionName,
            "--json"
        )
        if ($attached.schema -ne "colab.attach.v1" -or
            -not $attached.control_connected) {
            throw "Attach did not establish a control session"
        }
        $trackedLocally = $true
    }
    else {
        Invoke-Colab new --session $sessionName
        if ($LASTEXITCODE -ne 0) {
            throw "CPU session allocation failed"
        }
        $trackedLocally = $true
    }

    Invoke-Colab submit --session $sessionName --name $jobId -- `
        python -u -c `
        "import sys,time; print('MONITOR_STDOUT',flush=True); print('MONITOR_STDERR',file=sys.stderr,flush=True); time.sleep(2)"
    if ($LASTEXITCODE -ne 0) {
        throw "Job submission failed"
    }

    $summary = Invoke-ColabJson @(
        "monitor", $jobId,
        "--session", $sessionName,
        "--interval", "0.5",
        "--probe-every", "1",
        "--probe-timeout", "10",
        "--output", $outputRoot,
        "--json"
    )
    if ($summary.schema -ne "colab.monitor.summary.v1" -or
        $summary.exit_code -ne 0) {
        throw "Monitor did not report success"
    }

    $stdout = Get-Content -Raw -LiteralPath (
        Join-Path $outputRoot "stdout.log"
    )
    $stderr = Get-Content -Raw -LiteralPath (
        Join-Path $outputRoot "stderr.log"
    )
    if ($stdout -notmatch "MONITOR_STDOUT" -or
        $stderr -notmatch "MONITOR_STDERR") {
        throw "Local monitor evidence is incomplete"
    }
    foreach ($name in @(
        "job.jsonl",
        "resources.jsonl",
        "events.jsonl",
        "monitor_state.json",
        "summary.json"
    )) {
        if (-not (Test-Path -LiteralPath (
            Join-Path $outputRoot $name
        ))) {
            throw "Missing monitor artifact: $name"
        }
    }

    $networkDoctor = Invoke-ColabJson @(
        "doctor",
        "--json",
        "--network",
        "--timeout", "10"
    )
    if ($networkDoctor.network.status -ne "ok") {
        throw "Network doctor failed"
    }

    Write-Host "LIVE_CPU_OPERATIONAL_RECOVERY_OK" `
        -ForegroundColor Green
}
catch {
    $primaryError = $_
}
finally {
    if ($trackedLocally) {
        try {
            Invoke-Colab stop --session $sessionName
            if ($LASTEXITCODE -ne 0) {
                $cleanupErrors.Add(
                    "colab stop exited with $LASTEXITCODE"
                )
            }
        }
        catch {
            $cleanupErrors.Add("colab stop failed: $_")
        }
    }
    Remove-Item -LiteralPath $outputRoot `
        -Recurse -Force -ErrorAction SilentlyContinue
}

if ($cleanupErrors.Count -gt 0) {
    $message = $cleanupErrors -join "; "
    if ($null -ne $primaryError) {
        throw "$($primaryError.Exception.Message); cleanup: $message"
    }
    throw $message
}
if ($null -ne $primaryError) {
    throw $primaryError
}
