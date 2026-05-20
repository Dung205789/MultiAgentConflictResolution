import json
from pathlib import Path


def _lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


def build_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _lines(
                """
# ProjectMem Secondary Real-Model Runner

Notebook nay clone repo tu GitHub va chay secondary track `end_to_end_extract` voi model that.

- benchmark: `MemoryAgentBench / Conflict_Resolution`
- slice: scenario `1` trong 2 scenario dau tien
- mode: `conflict_aware`
- models: `Qwen/Qwen2.5-3B-Instruct` cho ca hai agent
- khong `--use-dummy`
- khong `--allow-structured-fallback-in-end-to-end`
- GPU chap nhan: `T4` uu tien, neu Kaggle cap `P100` thi notebook se cai lai PyTorch CUDA 12.6 tuong thich truoc khi chay
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "id": "bootstrap",
            "source": _lines(
                """
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/Dung205789/MultiAgentConflictResolution.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/ProjectMem"
OUTPUT_ROOT = "/kaggle/working/projectmem_secondary_real_qwen3b"
BOOTSTRAP_PATH = f"{OUTPUT_ROOT}/bootstrap_status.json"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def write_bootstrap(stage, extra=None):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    Path(BOOTSTRAP_PATH).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

write_bootstrap("bootstrap_start")
if os.path.isdir(WORKDIR):
    subprocess.run(["git", "-C", WORKDIR, "fetch", "origin"], check=False)
    subprocess.run(["git", "-C", WORKDIR, "checkout", BRANCH], check=False)
    subprocess.run(["git", "-C", WORKDIR, "pull", "origin", BRANCH], check=False)
else:
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, WORKDIR], check=True)

os.chdir(WORKDIR)
git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
print("WORKDIR", os.getcwd())
print("GIT_SHA", git_sha)
write_bootstrap("repo_synced", {"git_sha": git_sha})
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "id": "deps-gpu",
            "source": _lines(
                """
%cd /kaggle/working/ProjectMem
import json
import os
import subprocess
from pathlib import Path

os.environ["PIP_PROGRESS_BAR"] = "off"
!python -m pip install -q --progress-bar off --upgrade pip
!python -m pip install -q --progress-bar off -r requirements.txt
!python -m pip install -q --progress-bar off "transformers==4.49.0" "accelerate==1.3.0" "bitsandbytes>=0.43.0,<0.46.0"

gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True,
).splitlines()[0].strip()
print("GPU (nvidia-smi):", gpu_name)
accepted = ["T4", "P100"]
if not any(token in gpu_name for token in accepted):
    raise RuntimeError(f"Unsupported GPU '{gpu_name}'. Expected T4 or P100.")

if "P100" in gpu_name:
    print("P100 detected -> reinstalling PyTorch cu126 for sm_60 compatibility")
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "-q",
            "--progress-bar",
            "off",
            "--force-reinstall",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "torch",
            "torchvision",
            "torchaudio",
        ],
        check=True,
    )

hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("HF login complete")
else:
    print("HF_TOKEN not found; public downloads only")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Kaggle GPU is required for this notebook.")
gpu_name = torch.cuda.get_device_name(0)
print("GPU (torch):", gpu_name)
profile = {
    "gpu_name": gpu_name,
    "cuda_device_count": torch.cuda.device_count(),
    "accepted": accepted,
}
Path(f"{OUTPUT_ROOT}/gpu_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
write_bootstrap("deps_ready", profile)
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "id": "run-secondary-real",
            "source": _lines(
                """
%cd /kaggle/working/ProjectMem
import json
import subprocess
from pathlib import Path

OUTPUT_DIR = "/kaggle/working/projectmem_secondary_real_qwen3b"
cmd = [
    "python", "scripts/run_secondary_track_real_model.py",
    "--output-dir", OUTPUT_DIR,
    "--scenario-index", "1",
    "--agent1-model", "Qwen/Qwen2.5-3B-Instruct",
    "--agent2-model", "Qwen/Qwen2.5-3B-Instruct",
    "--device", "auto",
]
print("RUN:", " ".join(cmd))
subprocess.run(cmd, check=True)

report_path = Path(OUTPUT_DIR) / "custom_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "report_path": str(report_path),
        "results_keys": list(report.get("results", {}).keys()),
        "track_reporting": report.get("track_reporting", {}),
    }, ensure_ascii=False, indent=2))
else:
    print("Report not found yet:", report_path)
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "id": "zip-output",
            "source": _lines(
                """
%cd /kaggle/working
import shutil
from pathlib import Path

output_root = Path("/kaggle/working/projectmem_secondary_real_qwen3b")
zip_base = Path("/kaggle/working/projectmem_secondary_real_qwen3b_artifacts")
if output_root.exists():
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=str(output_root))
    print("ARCHIVE", archive)
else:
    print("Output root missing:", output_root)
                """
            ),
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    notebook = build_notebook()
    out_path = Path("kaggle/kaggle_runner_secondary_real_qwen3b.ipynb")
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    kernel_dir = Path("kaggle/kernels/projectmem_secondary_real_qwen3b")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kaggle_runner_secondary_real_qwen3b.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
