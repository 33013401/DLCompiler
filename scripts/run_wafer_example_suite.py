#!/usr/bin/env python3
"""Run every Wafer example file in a separate process and retain exact results.

Activate wafer-torch310 first. Kernel execution uses the installed wheel; the
example harness transports CPU reference storages through the real Kuiper API.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "third_party/wafer/examples"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution", choices=("compile", "hardware"), default="compile")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--maxfail", type=int, default=0)
    parser.add_argument("--select", nargs="*", help="Relative filenames; omitted means all 63 files")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(EXAMPLES.rglob("test_*.py"))
    if args.select:
        selected = set(args.select)
        files = [path for path in files if str(path.relative_to(EXAMPLES)) in selected]
        missing = selected - {str(path.relative_to(EXAMPLES)) for path in files}
        if missing:
            parser.error(f"Unknown example files: {sorted(missing)}")
    # Device assertion diagnostics and NOC demos follow ordinary numerical tests.
    files.sort(key=lambda path: (
        2 if path.name == "test_assert.py" or "tle" in path.parts
        else 1 if path.name == "test_dot_scaled.py" else 0, str(path)))
    environment = os.environ.copy()
    environment.update(DICP_BACKEND="wafer", USE_SIM_MODE="0", WAFER_ENABLE_RUNTIME="1",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    environment.setdefault("TRITON_CACHE_DIR", str(args.output_dir.parent / "cache"))
    blocked = None
    results = []
    for index, source in enumerate(files, 1):
        relative = str(source.relative_to(EXAMPLES))
        directory = args.output_dir / relative.removesuffix(".py")
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if args.resume and result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        if blocked:
            result = {"file": relative, "status": "blocked_after_device_error", "reason": blocked}
        else:
            events_path = directory / "events.jsonl"
            events_path.write_text("")
            environment["WAFER_EXAMPLE_EVENTS"] = str(events_path)
            command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-r", "a",
                       str(source), f"--wafer-execution={args.execution}",
                       f"--maxfail={args.maxfail}", f"--junitxml={directory / 'junit.xml'}"]
            started = time.monotonic()
            print(f"[{index}/{len(files)}] {args.execution}: {relative}", flush=True)
            timeout = False
            with (directory / "pytest.log").open("w") as output:
                process = subprocess.Popen(command, cwd=REPO.parent, env=environment,
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timeout = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = process.returncode
            events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
            tests = [event for event in events if event["event"] == "test_result"]
            counts = Counter(event["outcome"] for event in tests)
            nodes = next((event["nodeids"] for event in events if event["event"] == "collection"), [])
            launches = sum(event["event"] == "launch_complete" for event in events)
            started_launches = sum(event["event"] == "launch_start" for event in events)
            errors = [event for event in events if event["event"] == "launch_error"]
            status = "timeout" if timeout else "passed" if code == 0 else "failed"
            if args.execution == "compile" and code == 0:
                status = "compiled" if any(event["event"] == "compiled" for event in events) else "no_kernel_executed"
            result = {
                "file": relative, "execution": args.execution, "status": status,
                "returncode": code, "seconds": round(time.monotonic() - started, 3),
                "collected": len(nodes), "outcomes": dict(counts), "completed_launches": launches,
                "compiled_kernels": sum(event["event"] == "compiled" for event in events),
                "not_completed": sorted(set(nodes) - {event["nodeid"] for event in tests}),
                "command": command, "log": str(directory / "pytest.log"),
            }
            if args.execution == "hardware" and (errors or started_launches != launches):
                blocked = f"Device launch failed or did not finish in {relative}; inspect its events/log before further device work."
            print(f"  {status}: {dict(counts)}, launches={launches}, {result['seconds']}s", flush=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        summary = {"execution": args.execution, "files": results, "blocked_reason": blocked}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.output_dir / 'summary.json'}", flush=True)
    return int(any(result["status"] not in ("passed", "compiled") for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
