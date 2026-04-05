# vision_ui

Local offline web frontend for submitting image/video jobs to remote vision models over SSH.

Current MVP:
- create local jobs from uploaded files
- persist `job.json` and `status.json`
- preview uploaded inputs in the browser
- prepare a consistent structure for later SSH upload / remote execution / result download

Planned next:
- SSH/SCP transport layer
- remote `run_job.py` execution
- polling remote status and downloading results
- DUSt3R and MonST3R runners

Run locally:

```bash
uvicorn app:app --reload
```
