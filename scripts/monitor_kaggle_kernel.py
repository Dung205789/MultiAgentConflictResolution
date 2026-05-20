import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def kernel_status(kernel: str) -> str:
    completed = run(["kaggle", "kernels", "status", kernel], check=False)
    text = (completed.stdout or "") + (completed.stderr or "")
    for line in text.splitlines():
        if 'has status "' in line:
            return line.split('has status "', 1)[1].split('"', 1)[0]
    return text.strip() or f"UNKNOWN_EXIT_{completed.returncode}"


def maybe_download_output(kernel: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel, "-p", str(output_dir), "-o"], check=False)


def dump_state(output_dir: Path) -> None:
    candidates = [
        output_dir / "projectmem_secondary_real_qwen3b" / "custom_report.json",
        output_dir / "projectmem_secondary_real_qwen3b" / "custom_report.json.progress.json",
        output_dir / "projectmem_secondary_real_qwen3b" / "secondary_run.log",
        output_dir / "projectmem-secondary-real-qwen3b-t4p100.log",
    ]
    for path in candidates:
        if not path.exists():
            continue
        print(f"\n=== {path} ===")
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                print(json.dumps(payload, ensure_ascii=False, indent=2)[:6000])
            except Exception:
                print(path.read_text(encoding="utf-8", errors="replace")[-4000:])
        else:
            print(path.read_text(encoding="utf-8", errors="replace")[-4000:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll a Kaggle kernel and mirror outputs locally.")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=30)
    parser.add_argument("--kaggle-config-dir", default="")
    args = parser.parse_args()

    if args.kaggle_config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = args.kaggle_config_dir

    output_dir = Path(args.output_dir)
    last_status = None
    for idx in range(args.max_polls):
        status = kernel_status(args.kernel)
        if status != last_status:
            print(f"[poll {idx + 1}] status={status}")
            last_status = status
        maybe_download_output(args.kernel, output_dir)
        dump_state(output_dir)
        if any(token in status for token in ("COMPLETE", "ERROR", "CANCEL", "FAILED")):
            return 0
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
