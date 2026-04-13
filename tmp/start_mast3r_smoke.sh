#!/usr/bin/env bash
set -e
JOB=/hdd3/kykt26/jobs/mast3r_smoke_20260413
rm -rf "$JOB"
mkdir -p "$JOB/input" "$JOB/output" "$JOB/logs"
cp /hdd3/kykt26/code/mast3r/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg "$JOB/input/input_01.jpg"
cp /hdd3/kykt26/code/mast3r/assets/NLE_tower/91E9B685-7A7D-42D7-B933-23A800EE4129-83120-000041DAE12C8176.jpg "$JOB/input/input_02.jpg"
nohup conda run -n mast3r python /tmp/mast3r_runner.py --job-dir "$JOB" --model /hdd3/kykt26/code/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --repo /hdd3/kykt26/code/mast3r --image-size 512 --scene-graph complete --niter 300 --lr 0.01 --batch-size 1 --max-points 120000 --match-viz-count 30 > "$JOB/logs/runner.log" 2>&1 &
echo $! > "$JOB/runner.pid"
echo "STARTED $(cat "$JOB/runner.pid")"