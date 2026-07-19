param(
    [string]$Session = "windows-exec-control-smoke"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$notebook = Join-Path $PSScriptRoot "smoke.ipynb"
$outputNotebook = Join-Path $PSScriptRoot "smoke_output.ipynb"
$sessionName = "$Session-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
$primaryError = $null
$cleanupErrors = [System.Collections.Generic.List[string]]::new()

Push-Location $repoRoot
try {
    & uv run colab new --session $sessionName
    $newCode = $LASTEXITCODE
    if ($newCode -ne 0) {
        throw "CPU session allocation failed with exit code $newCode"
    }

    $selectedOutput = (& uv run colab exec --session $sessionName --file $notebook `
        --cell-title "Selected Cell" --fail-on-error 2>&1 | Out-String)
    $selectedCode = $LASTEXITCODE
    Write-Host $selectedOutput
    if ($selectedCode -ne 0) {
        throw "Selected-cell execution failed with exit code $selectedCode"
    }
    if ($selectedOutput -notmatch "WINDOWS_CELL_SELECTION_OK") {
        throw "Selected-cell success marker was not emitted"
    }
    if ($selectedOutput -match "UNSELECTED_CELL_EXECUTED") {
        throw "An unselected notebook cell was executed"
    }

    $errorOutput = (& uv run colab exec --session $sessionName --file $notebook `
        --cell-title "Error Cell" --fail-on-error 2>&1 | Out-String)
    $errorCode = $LASTEXITCODE
    Write-Host $errorOutput
    if ($errorCode -ne 1) {
        throw "Expected error-cell exit code 1, got $errorCode"
    }
    if ($errorOutput -notmatch "EXPECTED_FAIL_ON_ERROR") {
        throw "Expected remote error marker was not emitted"
    }

    Write-Host "LIVE_CPU_SMOKE_OK" -ForegroundColor Green
}
catch {
    $primaryError = $_
}
finally {
    try {
        & uv run colab stop --session $sessionName
        $stopCode = $LASTEXITCODE
        if ($stopCode -ne 0) {
            $cleanupErrors.Add("colab stop exited with code $stopCode")
        }
    }
    catch {
        $cleanupErrors.Add("colab stop failed: $_")
    }
    Remove-Item -LiteralPath $outputNotebook -Force -ErrorAction SilentlyContinue
    try {
        & uv run colab sessions
        $sessionsCode = $LASTEXITCODE
        if ($sessionsCode -ne 0) {
            $cleanupErrors.Add("colab sessions exited with code $sessionsCode")
        }
    }
    catch {
        $cleanupErrors.Add("colab sessions failed: $_")
    }
    Pop-Location
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
