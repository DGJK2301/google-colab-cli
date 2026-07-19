param(
    [string]$Session = "transfer-jobs-smoke",
    [string]$BundleRepo = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runId = [Guid]::NewGuid().ToString("N").Substring(0, 8)
$sessionName = "$Session-$runId"
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) "colab-cli-$runId"
$sourceFile = Join-Path $tempRoot "transfer-fixture.bin"
$downloadFile = Join-Path $tempRoot "transfer-roundtrip.bin"
$remoteFile = "content/transfer-fixture-$runId.bin"
$primaryError = $null
$cleanupErrors = [System.Collections.Generic.List[string]]::new()

New-Item -ItemType Directory -Path $tempRoot | Out-Null
if ($BundleRepo) {
    $bundlePath = (Resolve-Path $BundleRepo).Path
    $sourceFile = Join-Path $tempRoot "repository.bundle"
    $repoHead = (& git -C $bundlePath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $repoHead -notmatch "^[0-9a-f]{40}$") {
        throw "Unable to resolve BundleRepo HEAD"
    }
    & git -C $bundlePath bundle create $sourceFile --all
    if ($LASTEXITCODE -ne 0) {
        throw "git bundle create failed"
    }
}
else {
    $stream = [IO.File]::Open($sourceFile, [IO.FileMode]::CreateNew)
    try {
        $stream.SetLength(8MB)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
    $repoHead = $null
}
$sourceSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFile).Hash.ToLower()
$sourceSize = (Get-Item -LiteralPath $sourceFile).Length

Push-Location $repoRoot
try {
    & uv run colab new --session $sessionName
    if ($LASTEXITCODE -ne 0) {
        throw "CPU session allocation failed with exit code $LASTEXITCODE"
    }

    & uv run colab upload --session $sessionName --chunk-size-mib 1 `
        $sourceFile $remoteFile
    if ($LASTEXITCODE -ne 0) {
        throw "Chunked upload failed with exit code $LASTEXITCODE"
    }

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
    $verifyOutput = ($verifyCode | & uv run colab exec --session $sessionName `
        --fail-on-error 2>&1 | Out-String)
    Write-Host $verifyOutput
    if ($LASTEXITCODE -ne 0 -or $verifyOutput -notmatch "REMOTE_TRANSFER_SHA_OK") {
        throw "Remote size/SHA verification failed"
    }

    & uv run colab download --session $sessionName --chunk-size-mib 1 `
        $remoteFile $downloadFile
    if ($LASTEXITCODE -ne 0) {
        throw "Chunked download failed with exit code $LASTEXITCODE"
    }
    $downloadSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $downloadFile).Hash.ToLower()
    if ($downloadSha -ne $sourceSha) {
        throw "Downloaded SHA mismatch: $downloadSha != $sourceSha"
    }

    if ($repoHead) {
        $remoteBundle = "/$remoteFile".Replace("//", "/")
        & uv run colab submit --session $sessionName --name bundle-clone -- `
            git clone $remoteBundle /content/bundle-smoke-repo
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle clone job submission failed"
        }
        & uv run colab wait bundle-clone --session $sessionName --timeout 120
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle clone job failed"
        }
        & uv run colab submit --session $sessionName --name bundle-head -- `
            git -C /content/bundle-smoke-repo cat-file -e "$repoHead`^{commit}"
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle HEAD verification submission failed"
        }
        & uv run colab wait bundle-head --session $sessionName --timeout 60
        if ($LASTEXITCODE -ne 0) {
            throw "Exact bundle commit was not available remotely"
        }
    }

    & uv run colab submit --session $sessionName --name reconnect-smoke -- `
        python -u -c "import time; print('JOB_START', flush=True); time.sleep(3); print('JOB_END', flush=True)"
    if ($LASTEXITCODE -ne 0) {
        throw "Reconnect job submission failed"
    }
    Start-Sleep -Seconds 1
    $jobsOutput = (& uv run colab jobs --session $sessionName 2>&1 | Out-String)
    Write-Host $jobsOutput
    if ($LASTEXITCODE -ne 0 -or $jobsOutput -notmatch "reconnect-smoke") {
        throw "Persisted job was not listed"
    }
    $tailOutput = (& uv run colab tail reconnect-smoke --session $sessionName `
        --stream stdout --offset 0 2>&1 | Out-String)
    Write-Host $tailOutput
    if ($LASTEXITCODE -ne 0 -or $tailOutput -notmatch "JOB_START") {
        throw "Incremental tail did not recover the first log record"
    }
    $waitOutput = (& uv run colab wait reconnect-smoke --session $sessionName `
        --timeout 30 --poll-seconds 0.5 2>&1 | Out-String)
    Write-Host $waitOutput
    if ($LASTEXITCODE -ne 0 -or $waitOutput -notmatch "JOB_END") {
        throw "Reattached wait did not observe successful completion"
    }

    & uv run colab submit --session $sessionName --name cancel-smoke -- `
        python -u -c "import time; print('CANCEL_READY', flush=True); time.sleep(120)"
    if ($LASTEXITCODE -ne 0) {
        throw "Cancel job submission failed"
    }
    Start-Sleep -Seconds 1
    & uv run colab cancel cancel-smoke --session $sessionName --grace-seconds 2
    if ($LASTEXITCODE -ne 0) {
        throw "Remote cancel failed"
    }
    $cancelledOutput = (& uv run colab jobs --session $sessionName 2>&1 | Out-String)
    Write-Host $cancelledOutput
    if ($cancelledOutput -notmatch "cancel-smoke\s+cancelled") {
        throw "Cancelled state was not persisted"
    }

    Write-Host "LIVE_CPU_TRANSFER_JOBS_OK" -ForegroundColor Green
}
catch {
    $primaryError = $_
}
finally {
    try {
        & uv run colab stop --session $sessionName
        if ($LASTEXITCODE -ne 0) {
            $cleanupErrors.Add("colab stop exited with code $LASTEXITCODE")
        }
    }
    catch {
        $cleanupErrors.Add("colab stop failed: $_")
    }
    try {
        $sessionsOutput = (& uv run colab sessions 2>&1 | Out-String)
        Write-Host $sessionsOutput
        if ($LASTEXITCODE -ne 0) {
            $cleanupErrors.Add("colab sessions exited with code $LASTEXITCODE")
        }
        elseif ($sessionsOutput -notmatch "No active sessions found on server") {
            $cleanupErrors.Add("active Colab assignments remain after cleanup")
        }
    }
    catch {
        $cleanupErrors.Add("colab sessions failed: $_")
    }
    Pop-Location
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
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
