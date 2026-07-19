param(
    [string]$Session = "windows-exec-control-smoke"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$notebook = Join-Path $PSScriptRoot "smoke.ipynb"
$outputNotebook = Join-Path $PSScriptRoot "smoke_output.ipynb"
$sessionCreated = $false

Push-Location $repoRoot
try {
    & uv run colab new --session $Session
    $newCode = $LASTEXITCODE
    if ($newCode -ne 0) {
        throw "CPU session allocation failed with exit code $newCode"
    }
    $sessionCreated = $true

    $selectedOutput = (& uv run colab exec --session $Session --file $notebook `
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

    $errorOutput = (& uv run colab exec --session $Session --file $notebook `
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
finally {
    if ($sessionCreated) {
        & uv run colab stop --session $Session
    }
    Remove-Item -LiteralPath $outputNotebook -Force -ErrorAction SilentlyContinue
    & uv run colab sessions
    Pop-Location
}
