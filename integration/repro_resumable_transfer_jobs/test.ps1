param(
    [string]$Session = "transfer-jobs-smoke",
    [string]$BundleRepo = "",
    [ValidateSet(8, 64)]
    [int]$FileSizeMiB = 8,
    [double[]]$ChunkSizesMiB = @(0.25),
    [string]$ColabCommand = "colab",
    [string]$ResultPath = "",
    [switch]$SkipProcessLeakCheck
)

$ErrorActionPreference = "Stop"
$runId = [Guid]::NewGuid().ToString("N").Substring(0, 8)
$sessionName = "$Session-$runId"
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) "colab-cli-$runId"
$sourceFile = Join-Path $tempRoot "transfer-fixture.bin"
$primaryError = $null
$cleanupErrors = [System.Collections.Generic.List[string]]::new()
$benchmarkRows = [System.Collections.Generic.List[object]]::new()
$preservedRemote = $null

function Invoke-Colab {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $ColabCommand @Arguments
}

function Invoke-ColabJson {
    param([string[]]$Arguments)
    $stdout = (Invoke-Colab @Arguments | Out-String)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "colab $($Arguments -join ' ') failed with exit code $exitCode"
    }
    try {
        return $stdout | ConvertFrom-Json -Depth 32
    }
    catch {
        throw "Command returned invalid JSON: $stdout"
    }
}

function Assert-TransferResult {
    param(
        [object]$Payload,
        [string]$Direction,
        [long]$ExpectedSize,
        [string]$ExpectedSha
    )
    if ($Payload.schema -ne "colab.transfer.v1") {
        throw "Unexpected transfer schema: $($Payload.schema)"
    }
    if (-not $Payload.ok -or $Payload.status -ne "completed") {
        throw "Transfer did not complete: $($Payload | ConvertTo-Json -Depth 12)"
    }
    if ($Payload.direction -ne $Direction) {
        throw "Unexpected direction: $($Payload.direction)"
    }
    if ([long]$Payload.completed_bytes -ne $ExpectedSize -or
        [long]$Payload.total_bytes -ne $ExpectedSize) {
        throw "Transfer size mismatch in JSON result"
    }
    if ($Payload.sha256 -ne $ExpectedSha) {
        throw "Transfer SHA mismatch in JSON result"
    }
    if ($null -eq $Payload.elapsed_seconds -or
        $null -eq $Payload.retry_count) {
        throw "Transfer telemetry fields are missing"
    }
}

function Get-ColabProcessIds {
    if ($SkipProcessLeakCheck) {
        return @()
    }
    try {
        return @(
            Get-CimInstance Win32_Process |
                Where-Object {
                    $_.Name -ieq "colab.exe" -or
                    ($_.CommandLine -and
                     $_.CommandLine -match "colab_cli")
                } |
                ForEach-Object { [int]$_.ProcessId }
        )
    }
    catch {
        throw "Unable to inspect Colab process tree: $_"
    }
}

function Assert-NoNewColabProcesses {
    param([int[]]$ExpectedProcessIds)
    if ($SkipProcessLeakCheck) {
        return
    }
    Start-Sleep -Milliseconds 750
    $current = @(Get-ColabProcessIds)
    $unexpected = @(
        $current | Where-Object {
            $_ -notin $ExpectedProcessIds
        }
    )
    if ($unexpected.Count -gt 0) {
        throw "Unexpected Colab CLI processes remain: $($unexpected -join ', ')"
    }
}

function Convert-ChunkLabel {
    param([double]$Value)
    return ($Value.ToString(
        "0.############",
        [Globalization.CultureInfo]::InvariantCulture
    ) -replace "\.", "_")
}

New-Item -ItemType Directory -Path $tempRoot | Out-Null
if ($BundleRepo) {
    $bundlePath = (Resolve-Path $BundleRepo).Path
    $sourceFile = Join-Path $tempRoot "repository.bundle"
    $repoHead = (& git -C $bundlePath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or
        $repoHead -notmatch "^[0-9a-f]{40}$") {
        throw "Unable to resolve BundleRepo HEAD"
    }
    & git -C $bundlePath bundle create $sourceFile --all
    if ($LASTEXITCODE -ne 0) {
        throw "git bundle create failed"
    }
}
else {
    $stream = [IO.File]::Open(
        $sourceFile,
        [IO.FileMode]::CreateNew
    )
    try {
        $stream.SetLength($FileSizeMiB * 1MB)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
    $repoHead = $null
}

$sourceSha = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFile
).Hash.ToLower()
$sourceSize = (Get-Item -LiteralPath $sourceFile).Length

try {
    Invoke-Colab new --session $sessionName
    if ($LASTEXITCODE -ne 0) {
        throw "CPU session allocation failed with exit code $LASTEXITCODE"
    }
    $expectedProcessIds = @(Get-ColabProcessIds)

    foreach ($chunkSize in $ChunkSizesMiB) {
        if ([double]::IsNaN($chunkSize) -or
            [double]::IsInfinity($chunkSize) -or
            $chunkSize -le 0) {
            throw "Invalid chunk size: $chunkSize"
        }
        $chunkText = $chunkSize.ToString(
            "0.############",
            [Globalization.CultureInfo]::InvariantCulture
        )
        $label = Convert-ChunkLabel $chunkSize
        $remoteFile = "content/transfer-$runId-$label.bin"
        $downloadFile = Join-Path $tempRoot "roundtrip-$label.bin"

        $upload = Invoke-ColabJson @(
            "upload",
            "--session", $sessionName,
            "--chunk-size-mib", $chunkText,
            "--json",
            $sourceFile,
            $remoteFile
        )
        Assert-TransferResult `
            -Payload $upload `
            -Direction "upload" `
            -ExpectedSize $sourceSize `
            -ExpectedSha $sourceSha
        Assert-NoNewColabProcesses $expectedProcessIds

        $verifyCode = @"
import hashlib, os
path = '/$remoteFile'.replace('//', '/')
h = hashlib.sha256()
with open(path, 'rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
assert os.path.getsize(path) == $sourceSize
assert h.hexdigest() == '$sourceSha'
print('REMOTE_TRANSFER_SHA_OK')
"@
        $verifyOutput = (
            $verifyCode |
                Invoke-Colab exec --session $sessionName `
                    --fail-on-error 2>&1 |
                Out-String
        )
        if ($LASTEXITCODE -ne 0 -or
            $verifyOutput -notmatch "REMOTE_TRANSFER_SHA_OK") {
            throw "Remote size/SHA verification failed"
        }

        $download = Invoke-ColabJson @(
            "download",
            "--session", $sessionName,
            "--chunk-size-mib", $chunkText,
            "--json",
            $remoteFile,
            $downloadFile
        )
        Assert-TransferResult `
            -Payload $download `
            -Direction "download" `
            -ExpectedSize $sourceSize `
            -ExpectedSha $sourceSha
        Assert-NoNewColabProcesses $expectedProcessIds

        $downloadSha = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $downloadFile
        ).Hash.ToLower()
        if ($downloadSha -ne $sourceSha) {
            throw "Downloaded SHA mismatch: $downloadSha != $sourceSha"
        }

        $benchmarkRows.Add([pscustomobject]@{
            chunk_size_mib = $chunkSize
            source_size = $sourceSize
            upload_elapsed_seconds = $upload.elapsed_seconds
            upload_mib_per_second = $upload.mib_per_second
            upload_retry_count = $upload.retry_count
            download_elapsed_seconds = $download.elapsed_seconds
            download_mib_per_second = $download.mib_per_second
            download_retry_count = $download.retry_count
        })

        if ($repoHead -and $null -eq $preservedRemote) {
            $preservedRemote = $remoteFile
        }
        else {
            Invoke-Colab rm --session $sessionName $remoteFile
            if ($LASTEXITCODE -ne 0) {
                throw "Remote test file cleanup failed"
            }
        }
    }

    if ($repoHead) {
        $remoteBundle = "/$preservedRemote".Replace("//", "/")
        Invoke-Colab submit --session $sessionName `
            --name bundle-clone -- `
            git clone $remoteBundle /content/bundle-smoke-repo
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle clone job submission failed"
        }
        Invoke-Colab wait bundle-clone --session $sessionName --timeout 120
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle clone job failed"
        }
        Invoke-Colab submit --session $sessionName `
            --name bundle-head -- `
            git -C /content/bundle-smoke-repo `
                cat-file -e "$repoHead`^{commit}"
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle HEAD verification submission failed"
        }
        Invoke-Colab wait bundle-head --session $sessionName --timeout 60
        if ($LASTEXITCODE -ne 0) {
            throw "Exact bundle commit was not available remotely"
        }
        Invoke-Colab rm --session $sessionName $preservedRemote
    }

    Invoke-Colab submit --session $sessionName `
        --name reconnect-smoke -- `
        python -u -c `
            "import time; print('JOB_START', flush=True); time.sleep(3); print('JOB_END', flush=True)"
    if ($LASTEXITCODE -ne 0) {
        throw "Reconnect job submission failed"
    }
    Start-Sleep -Seconds 1
    $jobsOutput = (
        Invoke-Colab jobs --session $sessionName 2>&1 |
            Out-String
    )
    if ($LASTEXITCODE -ne 0 -or
        $jobsOutput -notmatch "reconnect-smoke") {
        throw "Persisted job was not listed"
    }
    $tailOutput = (
        Invoke-Colab tail reconnect-smoke `
            --session $sessionName `
            --stream stdout `
            --offset 0 2>&1 |
            Out-String
    )
    if ($LASTEXITCODE -ne 0 -or
        $tailOutput -notmatch "JOB_START") {
        throw "Incremental tail did not recover the first log record"
    }
    $waitOutput = (
        Invoke-Colab wait reconnect-smoke `
            --session $sessionName `
            --timeout 30 `
            --poll-seconds 0.5 2>&1 |
            Out-String
    )
    if ($LASTEXITCODE -ne 0 -or
        $waitOutput -notmatch "JOB_END") {
        throw "Reattached wait did not observe completion"
    }

    Invoke-Colab submit --session $sessionName `
        --name cancel-smoke -- `
        python -u -c `
            "import time; print('CANCEL_READY', flush=True); time.sleep(120)"
    if ($LASTEXITCODE -ne 0) {
        throw "Cancel job submission failed"
    }
    Start-Sleep -Seconds 1
    Invoke-Colab cancel cancel-smoke `
        --session $sessionName `
        --grace-seconds 2
    if ($LASTEXITCODE -ne 0) {
        throw "Remote cancel failed"
    }
    $cancelledOutput = (
        Invoke-Colab jobs --session $sessionName 2>&1 |
            Out-String
    )
    if ($cancelledOutput -notmatch "cancel-smoke\s+cancelled") {
        throw "Cancelled state was not persisted"
    }

    $summary = [pscustomobject]@{
        schema = "colab.transfer.benchmark.v1"
        session = $sessionName
        source_size = $sourceSize
        source_sha256 = $sourceSha
        results = @($benchmarkRows)
    }
    $summaryJson = $summary | ConvertTo-Json -Depth 12
    Write-Host $summaryJson
    if ($ResultPath) {
        $resultParent = Split-Path -Parent $ResultPath
        if ($resultParent) {
            New-Item -ItemType Directory `
                -Path $resultParent `
                -Force | Out-Null
        }
        Set-Content -LiteralPath $ResultPath `
            -Value $summaryJson `
            -Encoding utf8NoBOM
    }

    Write-Host "LIVE_CPU_TRANSFER_JOBS_OK" -ForegroundColor Green
}
catch {
    $primaryError = $_
}
finally {
    try {
        Invoke-Colab stop --session $sessionName
        if ($LASTEXITCODE -ne 0) {
            $cleanupErrors.Add(
                "colab stop exited with code $LASTEXITCODE"
            )
        }
    }
    catch {
        $cleanupErrors.Add("colab stop failed: $_")
    }
    try {
        $sessionsOutput = (
            Invoke-Colab sessions 2>&1 |
                Out-String
        )
        if ($LASTEXITCODE -ne 0) {
            $cleanupErrors.Add(
                "colab sessions exited with code $LASTEXITCODE"
            )
        }
        elseif ($sessionsOutput -notmatch
            "No active sessions found on server") {
            $cleanupErrors.Add(
                "active Colab assignments remain after cleanup"
            )
        }
    }
    catch {
        $cleanupErrors.Add("colab sessions failed: $_")
    }
    Remove-Item -LiteralPath $tempRoot `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue
}

if ($cleanupErrors.Count -gt 0) {
    $cleanupMessage = $cleanupErrors -join "; "
    if ($null -ne $primaryError) {
        throw "$($primaryError.Exception.Message); cleanup failed: $cleanupMessage"
    }
    throw "Cleanup failed: $cleanupMessage"
}
if ($null -ne $primaryError) {
    throw $primaryError
}
