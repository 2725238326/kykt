param(
    [string]$SourceDir = (Join-Path $PSScriptRoot "..\model_uploads\monst3r"),
    [string]$HostAlias = "KYKT-UI"
)

$ErrorActionPreference = "Stop"

$source = Resolve-Path -LiteralPath $SourceDir

$requiredFiles = @{
    "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth" = "/hdd3/kykt26/code/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
    "Tartan-C-T-TSKH-spring540x960-M.pth" = "/hdd3/kykt26/code/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
    "sam2.1_hiera_large.pt" = "/hdd3/kykt26/code/monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
}

$optionalFiles = @{
    "models.zip" = "/hdd3/kykt26/code/monst3r/third_party/RAFT/models.zip"
}

Write-Host "Checking local MonST3R weight files in $source"
foreach ($name in $requiredFiles.Keys) {
    $path = Join-Path $source $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $path"
    }
}

Write-Host "Preparing server directories..."
& ssh $HostAlias "mkdir -p /hdd3/kykt26/code/monst3r/checkpoints /hdd3/kykt26/code/monst3r/third_party/RAFT/models /hdd3/kykt26/code/monst3r/third_party/sam2/checkpoints"
if ($LASTEXITCODE -ne 0) { throw "Failed to prepare remote directories." }

foreach ($entry in $requiredFiles.GetEnumerator()) {
    $localPath = Join-Path $source $entry.Key
    $remotePath = $entry.Value
    Write-Host "Uploading $($entry.Key) -> $remotePath"
    & scp $localPath "${HostAlias}:$remotePath"
    if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($entry.Key)" }
}

foreach ($entry in $optionalFiles.GetEnumerator()) {
    $localPath = Join-Path $source $entry.Key
    if (Test-Path -LiteralPath $localPath) {
        $remotePath = $entry.Value
        Write-Host "Uploading optional $($entry.Key) -> $remotePath"
        & scp $localPath "${HostAlias}:$remotePath"
        if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($entry.Key)" }

        Write-Host "Unpacking optional RAFT models.zip on server..."
        & ssh $HostAlias "cd /hdd3/kykt26/code/monst3r/third_party/RAFT && unzip -o models.zip >/tmp/kykt_unzip_raft_models.log && tail -n 20 /tmp/kykt_unzip_raft_models.log"
        if ($LASTEXITCODE -ne 0) { throw "Failed to unzip models.zip on server." }
    }
}

Write-Host "Verifying remote files..."
& ssh $HostAlias "set -e; ls -lah /hdd3/kykt26/code/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth /hdd3/kykt26/code/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth /hdd3/kykt26/code/monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt; find /hdd3/kykt26/code/monst3r/third_party/RAFT/models -maxdepth 1 -type f -printf '%f %k KB\n' | sort"
if ($LASTEXITCODE -ne 0) { throw "Remote verification failed." }

Write-Host "MonST3R weight upload complete."
