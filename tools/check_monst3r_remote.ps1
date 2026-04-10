param(
    [string]$HostAlias = "KYKT-UI"
)

$ErrorActionPreference = "Stop"

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

$remoteScript = @'
set -euo pipefail

REPO="/hdd3/kykt26/code/monst3r"
MAIN_WEIGHT="$REPO/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
RAFT_WEIGHT="$REPO/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
SAM2_WEIGHT="$REPO/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"

echo "===== MonST3R remote check ====="
echo "host: $(hostname)"
echo "pwd:  $(pwd)"
echo

echo "===== repo ====="
if [ -d "$REPO" ]; then
  echo "repo ok: $REPO"
else
  echo "repo missing: $REPO"
fi

if [ -f "$REPO/demo.py" ]; then
  echo "demo.py ok"
else
  echo "demo.py missing"
fi
echo

echo "===== env ====="
if conda env list | awk '{print $1}' | grep -qx "monst3r"; then
  echo "conda env ok: monst3r"
else
  echo "conda env missing: monst3r"
fi

conda run -n monst3r python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"
echo

echo "===== weights ====="
for path in "$MAIN_WEIGHT" "$RAFT_WEIGHT" "$SAM2_WEIGHT"; do
  if [ -f "$path" ]; then
    ls -lh "$path"
  else
    echo "missing: $path"
  fi
done
echo

echo "===== demo import smoke test ====="
cd "$REPO"
conda run -n monst3r python demo.py --help >/tmp/kykt_monst3r_demo_help.txt
tail -n 15 /tmp/kykt_monst3r_demo_help.txt
echo

echo "===== suggested next command ====="
echo "cd /hdd3/kykt26/code/monst3r"
echo "conda activate monst3r"
echo "python demo.py --input_dir /hdd3/kykt26/<your_input_dir> --output_dir demo_tmp --seq_name <seq_name> --weights checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth --silent"
'@

Write-Host "Checking MonST3R remote status on $HostAlias ..."
Invoke-RemoteBash -Script $remoteScript
