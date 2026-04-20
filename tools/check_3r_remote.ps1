param(
    [Alias("Alias", "HostAlias")]
    [string]$SshAlias = "KYKT-UI",
    [switch]$Json
)

$ErrorActionPreference = "Stop"

function Invoke-RemoteBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script
    )

    $bom = [string][char]0xFEFF
    $cleanScript = ($Script -replace [regex]::Escape($bom), "") -replace "`r", ""
    $tmp = New-TemporaryFile
    try {
        [System.IO.File]::WriteAllText($tmp.FullName, $cleanScript, [System.Text.UTF8Encoding]::new($false))
        $cmd = 'type "{0}" | ssh {1} "bash -s --"' -f $tmp.FullName, $SshAlias
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "Remote command failed with exit code $LASTEXITCODE."
        }
    } finally {
        Remove-Item -LiteralPath $tmp.FullName -Force -ErrorAction SilentlyContinue
    }
}

$remoteScript = @'
set -uo pipefail

ROOT="/hdd3/kykt26/code"
REPOS=(mast3r monst3r spann3r align3r fast3r cut3r)

missing_dirs=0
missing_envs=0
missing_required=0
warn_count=0

line() {
  printf '%*s\n' "${1:-100}" '' | tr ' ' '-'
}

section() {
  printf '\n%s\n' "$1"
  line 100
}

status() {
  local kind="$1"
  local path="$2"
  if [ "$kind" = "dir" ]; then
    [ -d "$path" ] && printf 'OK' || printf 'MISSING'
  else
    [ -f "$path" ] && printf 'OK' || printf 'MISSING'
  fi
}

size_of() {
  local path="$1"
  if [ -e "$path" ]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    printf '-'
  fi
}

mtime_of() {
  local path="$1"
  if [ -e "$path" ]; then
    stat -c '%y' "$path" 2>/dev/null | cut -d'.' -f1
  else
    printf '-'
  fi
}

repo_state() {
  local repo="$1"
  if [ ! -d "$repo" ]; then
    printf 'MISSING'
    return
  fi

  local files
  files=$(find "$repo" -mindepth 1 -maxdepth 2 -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$files" = "1" ] && [ -f "$repo/README_SETUP.md" ]; then
    printf 'PLANNED'
  else
    printf 'OK'
  fi
}

conda_envs() {
  conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}'
}

env_exists() {
  local env="$1"
  printf '%s\n' "$CONDA_ENVS" | grep -qx "$env"
}

env_path() {
  local env="$1"
  printf '%s\n' "$CONDA_ENV_LIST" | awk -v e="$env" '$1 == e {print $NF; found=1} END {if (!found) print "-"}'
}

file_status_for_summary() {
  local severity="$1"
  local path="$2"
  if [ ! -f "$path" ]; then
    if [ "$severity" = "required" ]; then
      missing_required=$((missing_required + 1))
    else
      warn_count=$((warn_count + 1))
    fi
  fi
}

printf 'Remote 3R deployment check\n'
printf 'Host:      %s\n' "$(hostname 2>/dev/null || printf unknown)"
printf 'Root:      %s\n' "$ROOT"
printf 'Checked:   %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"

section "Directories"
printf '%-10s %-9s %-9s %-7s %-19s %s\n' "Component" "State" "README" "Size" "Updated" "Path"
line 100
for repo_name in "${REPOS[@]}"; do
  repo="$ROOT/$repo_name"
  state=$(repo_state "$repo")
  readme=$(status file "$repo/README_SETUP.md")
  size=$(size_of "$repo")
  updated=$(mtime_of "$repo")
  printf '%-10s %-9s %-9s %-7s %-19s %s\n' "$repo_name" "$state" "$readme" "$size" "$updated" "$repo"
  [ "$state" = "MISSING" ] && missing_dirs=$((missing_dirs + 1))
  [ "$readme" = "MISSING" ] && warn_count=$((warn_count + 1))
done

section "README_SETUP.md"
printf '%-10s %-9s %-12s %s\n' "Component" "Status" "SetupState" "Path"
line 100
for repo_name in "${REPOS[@]}"; do
  readme="$ROOT/$repo_name/README_SETUP.md"
  if [ -f "$readme" ]; then
    setup_state=$(grep -m1 -E '^Status:' "$readme" 2>/dev/null | sed -E 's/^Status:[[:space:]]*//')
    [ -z "$setup_state" ] && setup_state="-"
    printf '%-10s %-9s %-12s %s\n' "$repo_name" "OK" "$setup_state" "$readme"
  else
    printf '%-10s %-9s %-12s %s\n' "$repo_name" "MISSING" "-" "$readme"
  fi
done

CONDA_ENV_LIST="$(conda env list 2>/dev/null || true)"
CONDA_ENVS="$(printf '%s\n' "$CONDA_ENV_LIST" | awk 'NF && $1 !~ /^#/ {print $1}')"

section "Conda environments"
printf '%-10s %-12s %-9s %s\n' "Component" "Env" "Status" "Path"
line 100
while IFS='|' read -r component env note; do
  [ -z "$component" ] && continue
  if env_exists "$env"; then
    printf '%-10s %-12s %-9s %s\n' "$component" "$env" "OK" "$(env_path "$env")"
  else
    printf '%-10s %-12s %-9s %s\n' "$component" "$env" "MISSING" "$note"
    missing_envs=$((missing_envs + 1))
  fi
done <<'ENV_ROWS'
mast3r|mast3r|dedicated env expected
monst3r|monst3r|dedicated env expected
spann3r|spann3r|dedicated env expected by README_SETUP.md
align3r|align3r|dedicated env expected by README_SETUP.md
fast3r|fast3r|dedicated env expected by README_SETUP.md
cut3r|cut3r|dedicated env expected by README_SETUP.md
shared|dust3r|DUSt3R-family shared env
platform|kykt|KYKT platform env
sfm|sfm|SfM helper env
ENV_ROWS

section "Known weights and key files"
printf '%-10s %-12s %-9s %-8s %-7s %s\n' "Component" "Kind" "Need" "Status" "Size" "RelativePath"
line 100
while IFS='|' read -r component kind need relpath; do
  [ -z "$component" ] && continue
  path="$ROOT/$component/$relpath"
  st=$(status file "$path")
  size=$(size_of "$path")
  printf '%-10s %-12s %-9s %-8s %-7s %s\n' "$component" "$kind" "$need" "$st" "$size" "$relpath"
  file_status_for_summary "$need" "$path"
done <<'FILE_ROWS'
mast3r|readme|required|README.md
mast3r|entry|required|demo.py
mast3r|deps|required|requirements.txt
mast3r|weight|required|checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
monst3r|readme|required|README.md
monst3r|entry|required|demo.py
monst3r|deps|required|requirements.txt
monst3r|weight|required|checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth
monst3r|weight|required|third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth
monst3r|weight|required|third_party/sam2/checkpoints/sam2.1_hiera_large.pt
spann3r|setup|required|README_SETUP.md
align3r|setup|required|README_SETUP.md
fast3r|setup|required|README_SETUP.md
cut3r|setup|required|README_SETUP.md
FILE_ROWS

section "Discovered checkpoint-like files"
printf '%-10s %-7s %s\n' "Component" "Size" "RelativePath"
line 100
found_weight=0
for repo_name in "${REPOS[@]}"; do
  repo="$ROOT/$repo_name"
  [ -d "$repo" ] || continue
  while IFS= read -r -d '' file; do
    rel="${file#$repo/}"
    printf '%-10s %-7s %s\n' "$repo_name" "$(size_of "$file")" "$rel"
    found_weight=1
  done < <(find "$repo" -maxdepth 8 -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.safetensors' \) -print0 2>/dev/null | sort -z)
done
[ "$found_weight" = "0" ] && printf '%-10s %-7s %s\n' "-" "-" "none found"

section "Summary"
printf '%-24s %s\n' "Missing directories:" "$missing_dirs"
printf '%-24s %s\n' "Missing conda envs:" "$missing_envs"
printf '%-24s %s\n' "Missing required files:" "$missing_required"
printf '%-24s %s\n' "Warnings:" "$warn_count"

if [ "$missing_dirs" -eq 0 ] && [ "$missing_required" -eq 0 ]; then
  printf '\nResult: core directory/file checks passed. Review missing envs and README warnings above.\n'
else
  printf '\nResult: attention needed. See MISSING rows above.\n'
fi
'@

if ($Json) {
    $remoteScript = @'
python3 - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/hdd3/kykt26/code")
REPOS = ["mast3r", "monst3r", "spann3r", "align3r", "fast3r", "cut3r"]
KNOWN_FILES = [
    ("mast3r", "readme", "required", "README.md"),
    ("mast3r", "entry", "required", "demo.py"),
    ("mast3r", "deps", "required", "requirements.txt"),
    ("mast3r", "weight", "required", "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
    ("monst3r", "readme", "required", "README.md"),
    ("monst3r", "entry", "required", "demo.py"),
    ("monst3r", "deps", "required", "requirements.txt"),
    ("monst3r", "weight", "required", "checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"),
    ("monst3r", "weight", "required", "third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"),
    ("monst3r", "weight", "required", "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"),
    ("spann3r", "setup", "required", "README_SETUP.md"),
    ("align3r", "setup", "required", "README_SETUP.md"),
    ("fast3r", "setup", "required", "README_SETUP.md"),
    ("cut3r", "setup", "required", "README_SETUP.md"),
]
EXPECTED_ENVS = [
    ("mast3r", "mast3r"),
    ("monst3r", "monst3r"),
    ("spann3r", "spann3r"),
    ("align3r", "align3r"),
    ("fast3r", "fast3r"),
    ("cut3r", "cut3r"),
    ("shared", "dust3r"),
    ("platform", "kykt"),
    ("sfm", "sfm"),
]


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def size_bytes(path: Path) -> int | None:
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        total += (Path(root) / name).stat().st_size
                    except OSError:
                        pass
            return total
    except OSError:
        return None
    return None


def repo_state(path: Path) -> str:
    if not path.is_dir():
        return "missing"
    files = [p for p in path.glob("*") if p.is_file()]
    dirs = [p for p in path.glob("*") if p.is_dir()]
    if len(files) == 1 and files[0].name == "README_SETUP.md" and not dirs:
        return "planned"
    return "ready"


def parse_envs() -> dict[str, str]:
    envs: dict[str, str] = {}
    for line in run_text(["conda", "env", "list"]).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            envs[parts[0]] = parts[-1]
    return envs


envs = parse_envs()
directories = []
for name in REPOS:
    path = ROOT / name
    readme = path / "README_SETUP.md"
    directories.append(
        {
            "name": name,
            "path": str(path),
            "state": repo_state(path),
            "exists": path.is_dir(),
            "readme_setup": readme.is_file(),
            "size_bytes": size_bytes(path),
        }
    )

env_rows = [
    {"component": component, "env": env, "exists": env in envs, "path": envs.get(env)}
    for component, env in EXPECTED_ENVS
]

files = []
for component, kind, need, rel in KNOWN_FILES:
    path = ROOT / component / rel
    files.append(
        {
            "component": component,
            "kind": kind,
            "need": need,
            "relative_path": rel,
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": size_bytes(path),
        }
    )

checkpoints = []
for name in REPOS:
    repo = ROOT / name
    if not repo.is_dir():
        continue
    for path in repo.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}:
            checkpoints.append(
                {
                    "component": name,
                    "relative_path": str(path.relative_to(repo)),
                    "size_bytes": size_bytes(path),
                }
            )

summary = {
    "missing_directories": sum(1 for item in directories if not item["exists"]),
    "missing_conda_envs": sum(1 for item in env_rows if not item["exists"]),
    "missing_required_files": sum(1 for item in files if item["need"] == "required" and not item["exists"]),
    "warnings": sum(1 for item in directories if not item["readme_setup"]),
}
summary["ok"] = summary["missing_directories"] == 0 and summary["missing_required_files"] == 0

print(
    json.dumps(
        {
            "host": run_text(["hostname"]).strip() or None,
            "root": str(ROOT),
            "directories": directories,
            "conda_envs": env_rows,
            "known_files": files,
            "checkpoints": checkpoints,
            "summary": summary,
        },
        ensure_ascii=False,
        indent=2,
    )
)
PY
'@
}

if (-not $Json) {
    Write-Host "Checking remote 3R deployment on $SshAlias ..."
}
Invoke-RemoteBash -Script $remoteScript
