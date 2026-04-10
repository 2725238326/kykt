param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteInputPath,

    [Parameter(Mandatory = $true)]
    [string]$SeqName,

    [string]$HostAlias = "KYKT-UI",
    [string]$RemoteRepo = "/hdd3/kykt26/code/monst3r",
    [string]$OutputDir = "demo_tmp",
    [int]$BatchSize = 16,
    [switch]$NotBatchify,
    [switch]$RealTime,
    [switch]$WindowWise,
    [int]$WindowSize = 100,
    [double]$WindowOverlapRatio = 0.5
)

$ErrorActionPreference = "Stop"

function Convert-ToBashDoubleQuoted {
    param([string]$Value)
    $escaped = $Value.Replace('\', '\\').Replace('"', '\"').Replace('$', '\$').Replace('`', '\`')
    return '"' + $escaped + '"'
}

function Invoke-RemoteBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script
    )

    $Script = $Script -replace [string][char]0xFEFF, ""
    $tmp = New-TemporaryFile
    try {
        [System.IO.File]::WriteAllText($tmp.FullName, $Script, [System.Text.UTF8Encoding]::new($false))
        cmd /c "type `"$($tmp.FullName)`" | ssh $HostAlias bash -s --"
        if ($LASTEXITCODE -ne 0) {
            throw "Remote command failed with exit code $LASTEXITCODE."
        }
    } finally {
        Remove-Item -LiteralPath $tmp.FullName -Force -ErrorAction SilentlyContinue
    }
}

$repoQ = Convert-ToBashDoubleQuoted $RemoteRepo
$inputQ = Convert-ToBashDoubleQuoted $RemoteInputPath
$seqQ = Convert-ToBashDoubleQuoted $SeqName
$outputQ = Convert-ToBashDoubleQuoted $OutputDir

$extraArgs = @("--batch_size $BatchSize")
if ($NotBatchify) {
    $extraArgs += "--not_batchify"
}
if ($RealTime) {
    $extraArgs += "--real_time"
}
if ($WindowWise) {
    $extraArgs += "--window_wise"
    $extraArgs += "--window_size $WindowSize"
    $extraArgs += "--window_overlap_ratio $WindowOverlapRatio"
}
$extraArgString = $extraArgs -join " "

$remoteScript = @"
set -euo pipefail

REPO=$repoQ
INPUT_DIR=$inputQ
SEQ_NAME=$seqQ
OUTPUT_DIR=$outputQ
WEIGHT_PATH="\$REPO/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"

if [ ! -d "\$REPO" ]; then
  echo "MonST3R repo missing: \$REPO" >&2
  exit 1
fi

if [ ! -e "\$INPUT_DIR" ]; then
  echo "Remote input path missing: \$INPUT_DIR" >&2
  exit 1
fi

if [ ! -f "\$WEIGHT_PATH" ]; then
  echo "Main MonST3R weight missing: \$WEIGHT_PATH" >&2
  exit 1
fi

cd "\$REPO"
echo "Running MonST3R demo on \$(hostname)"
echo "input:  \$INPUT_DIR"
echo "output: \$OUTPUT_DIR/\$SEQ_NAME"
echo "repo:   \$REPO"

conda run -n monst3r python demo.py \
  --input_dir "\$INPUT_DIR" \
  --output_dir "\$OUTPUT_DIR" \
  --seq_name "\$SEQ_NAME" \
  --weights "\$WEIGHT_PATH" \
  --silent \
  $extraArgString

echo
echo "MonST3R demo finished. Output directory:"
echo "\$REPO/\$OUTPUT_DIR/\$SEQ_NAME"
find "\$REPO/\$OUTPUT_DIR/\$SEQ_NAME" -maxdepth 2 -type f | sort
"@

Write-Host "Launching MonST3R demo on $HostAlias ..."
Invoke-RemoteBash -Script $remoteScript
