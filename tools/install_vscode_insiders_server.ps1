param(
    [string]$Commit = "a4f5119796984330f866b3dd6b2ce0694ff5c814",
    [string]$Archive = "",
    [string]$HostAlias = "KYKT-UI",
    [string]$ExpectedSha256 = "3c50825217ce19689d05923a5a44672086bcae653eb61054d6a963fa721fab90"
)

$ErrorActionPreference = "Stop"

if (-not $Archive) {
    $Archive = "E:\kykt\vscode-server-insiders-$Commit.tar.gz"
}

if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Archive not found: $Archive"
}

Write-Host "Checking archive integrity..."
& tar -tzf $Archive > $null
if ($LASTEXITCODE -ne 0) {
    throw "Archive is not a complete tar.gz: $Archive"
}

$actualSha = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ExpectedSha256 -and $actualSha -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "SHA256 mismatch. Expected $ExpectedSha256 but got $actualSha"
}

Write-Host "Uploading VS Code Insiders server archive..."
& scp $Archive "${HostAlias}:/tmp/vscode-server-insiders-$Commit.tar.gz"
if ($LASTEXITCODE -ne 0) {
    throw "scp upload failed."
}

$remoteCmd = @"
set -e
COMMIT="$Commit"
ROOT="\$HOME/.vscode-server-insiders/bin/\$COMMIT"
rm -rf "\$ROOT"
mkdir -p "\$ROOT"
tar -xzf "/tmp/vscode-server-insiders-\$COMMIT.tar.gz" -C "\$ROOT" --strip-components=1
rm -f "/tmp/vscode-server-insiders-\$COMMIT.tar.gz"
touch "\$ROOT/0"
test -x "\$ROOT/node"
test -x "\$ROOT/bin/code-server"
echo "Installed VS Code Insiders server at \$ROOT"
"@

Write-Host "Installing on server..."
$remoteCmd | ssh $HostAlias bash
if ($LASTEXITCODE -ne 0) {
    throw "Remote install failed."
}

Write-Host "Done. Reconnect VS Code Insiders to $HostAlias."
