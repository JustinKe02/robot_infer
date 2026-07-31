#!/usr/bin/env python

"""Measure steady-state PI0.5 training memory for selected modes and batch sizes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SMOKE_SCRIPT = SCRIPT_DIR / "smoke_train.sh"
MEMORY_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
]
OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "NVML_SUCCESS == r",
)
STEP_PATTERN = re.compile(
    r"step:(?P<step>\d+).*?loss:(?P<loss>[0-9.e+-]+).*?"
    r"updt_s:(?P<update>[0-9.]+).*?data_s:(?P<data>[0-9.]+)"
)
PARAM_PATTERN = re.compile(r"num_learnable_params=(?P<params>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "expert", "lora"), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--poll-ms", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    return parser.parse_args()


def query_memory() -> dict[str, int]:
    output = subprocess.check_output(MEMORY_QUERY, text=True).strip().splitlines()[0]
    total, used, free, utilization = (int(value.strip()) for value in output.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free, "utilization_pct": utilization}


def wait_for_memory_release(target_used_mib: int, timeout_s: float = 120.0) -> dict[str, int]:
    deadline = time.monotonic() + timeout_s
    while True:
        current = query_memory()
        if current["used_mib"] <= target_used_mib + 256:
            return current
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"GPU memory did not return near baseline: current={current['used_mib']} MiB "
                f"baseline={target_used_mib} MiB"
            )
        time.sleep(1.0)


def parse_monitor(path: Path) -> list[dict[str, int]]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not all(part.isdigit() for part in parts):
            continue
        total, used, free, utilization = (int(part) for part in parts)
        samples.append(
            {
                "total_mib": total,
                "used_mib": used,
                "free_mib": free,
                "utilization_pct": utilization,
            }
        )
    return samples


def parse_training_log(path: Path, batch_size: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    step_matches = [match.groupdict() for match in STEP_PATTERN.finditer(text)]
    param_match = PARAM_PATTERN.search(text)
    last_step = step_matches[-1] if step_matches else None
    result: dict[str, Any] = {
        "oom_detected": any(marker in text for marker in OOM_MARKERS),
        "completed_steps": len(step_matches),
        "trainable_params": int(param_match.group("params")) if param_match else None,
        "last_loss": float(last_step["loss"]) if last_step else None,
        "last_update_s": float(last_step["update"]) if last_step else None,
        "last_data_s": float(last_step["data"]) if last_step else None,
    }
    if last_step:
        update_s = result["last_update_s"]
        end_to_end_s = update_s + result["last_data_s"]
        result["samples_per_update_second"] = batch_size / update_s
        result["samples_per_end_to_end_second"] = batch_size / end_to_end_s
    else:
        result["samples_per_update_second"] = None
        result["samples_per_end_to_end_second"] = None
    return result


def run_probe(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_id = f"{args.mode}_b{batch_size}"
    training_log = output_dir / f"{probe_id}.log"
    monitor_log = output_dir / f"{probe_id}_memory.csv"

    baseline = wait_for_memory_release(query_memory()["used_mib"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    environment = os.environ.copy()
    environment.update(
        {
            "FINETUNE_MODE": args.mode,
            "BATCH_SIZE": str(batch_size),
            "STEPS_OVERRIDE": str(args.steps),
            "RUN_STAMP": f"batch_benchmark_{timestamp}_{probe_id}",
            "LORA_R": str(args.lora_r),
        }
    )

    print(
        f"[jz/pi05/batch-benchmark] START mode={args.mode} batch={batch_size} "
        f"baseline={baseline['used_mib']}MiB",
        flush=True,
    )
    started = time.monotonic()
    with monitor_log.open("w", encoding="utf-8") as monitor_stream, training_log.open(
        "w", encoding="utf-8"
    ) as training_stream:
        monitor = subprocess.Popen(
            [*MEMORY_QUERY, "-lms", str(args.poll_ms)],
            stdout=monitor_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            completed = subprocess.run(
                ["bash", str(SMOKE_SCRIPT)],
                cwd=SCRIPT_DIR.parent.parent.parent,
                env=environment,
                stdout=training_stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()

    monitor_samples = parse_monitor(monitor_log)
    log_metrics = parse_training_log(training_log, batch_size)
    peak_used = max((sample["used_mib"] for sample in monitor_samples), default=baseline["used_mib"])
    minimum_free = min((sample["free_mib"] for sample in monitor_samples), default=baseline["free_mib"])
    success = completed.returncode == 0 and log_metrics["completed_steps"] == args.steps
    result = {
        "mode": args.mode,
        "batch_size": batch_size,
        "requested_steps": args.steps,
        "success": success,
        "return_code": completed.returncode,
        "baseline_used_mib": baseline["used_mib"],
        "peak_used_mib": peak_used,
        "training_increment_mib": peak_used - baseline["used_mib"],
        "minimum_free_mib": minimum_free,
        "wall_time_s": time.monotonic() - started,
        "training_log": str(training_log),
        "memory_log": str(monitor_log),
        **log_metrics,
    }
    status = "PASS" if success else "OOM" if result["oom_detected"] else "FAIL"
    print(
        f"[jz/pi05/batch-benchmark] {status} mode={args.mode} batch={batch_size} "
        f"peak={peak_used}MiB free={minimum_free}MiB steps={result['completed_steps']} "
        f"update_s={result['last_update_s']}",
        flush=True,
    )
    wait_for_memory_release(baseline["used_mib"])
    return result


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(results[0]) if results else []
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[jz/pi05/batch-benchmark] RESULTS json={json_path} csv={csv_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("At least 2 steps are required to measure steady-state optimizer memory")
    if args.poll_ms < 50:
        raise ValueError("poll-ms must be at least 50")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Batch sizes must be positive")

    results = []
    try:
        for batch_size in args.batch_sizes:
            results.append(run_probe(args, batch_size))
            write_results(args.output_dir, results)
    finally:
        if results:
            write_results(args.output_dir, results)


if __name__ == "__main__":
    main()
