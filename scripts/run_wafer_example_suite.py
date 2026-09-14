#!/usr/bin/env python3
"""Run Wafer examples or unchanged Ascend files and retain exact results.

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
    parser.add_argument("--suite", choices=("examples", "ascend"), default="examples")
    parser.add_argument("--select", nargs="*", help="Relative filenames; omitted means all test files")
    parser.add_argument("--nodeids-file", type=Path,
                        help="Ascend only: JSON array of exact original pytest nodeids to rerun")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = EXAMPLES if args.suite == "examples" else REPO / "test/ascend/passed_tests"
    files = sorted(path for path in source_root.rglob("test_*.py")
                   if path.name != "test_common.py")
    if args.nodeids_file:
        if args.suite != "ascend":
            parser.error("--nodeids-file requires --suite ascend")
        args.nodeids_file = args.nodeids_file.resolve()
        nodeids = json.loads(args.nodeids_file.read_text())
        if not isinstance(nodeids, list) or not nodeids or not all(isinstance(node, str) for node in nodeids):
            parser.error("--nodeids-file must contain a nonempty JSON array of nodeids")
        requested_files = {node.split("::", 1)[0] for node in nodeids}
        missing = requested_files - {str(path.relative_to(REPO)) for path in files}
        if missing:
            parser.error(f"Nodeids reference unknown Ascend files: {sorted(missing)}")
        files = [path for path in files if str(path.relative_to(REPO)) in requested_files]
    if args.select:
        selected = set(args.select)
        files = [path for path in files if str(path.relative_to(source_root)) in selected]
        missing = selected - {str(path.relative_to(source_root)) for path in files}
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
    if args.suite == "ascend":
        # Opt-in pytest plugin maps only the host tensor/reference entry points.
        # Original kernels, parameter sets and assertions remain unmodified.
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(REPO / "test/wafer"), environment.get("PYTHONPATH"))))
    blocked = None
    results = []
    for index, source in enumerate(files, 1):
        relative = str(source.relative_to(source_root))
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
            if args.suite == "ascend":
                command += ["-p", "upstream_adapter"]
                if args.nodeids_file:
                    command += [f"--wafer-nodeids={args.nodeids_file}"]
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
                "collection_errors": sum(event["event"] == "collection_error" for event in events),
                "compiled_kernels": sum(event["event"] == "compiled" for event in events),
                "not_completed": sorted(set(nodes) - {event["nodeid"] for event in tests}),
                "command": command, "log": str(directory / "pytest.log"),
            }
            if args.execution == "hardware" and (errors or started_launches != launches):
                blocked = f"Device launch failed or did not finish in {relative}; inspect its events/log before further device work."
            print(f"  {status}: {dict(counts)}, launches={launches}, {result['seconds']}s", flush=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        summary = {"suite": args.suite, "execution": args.execution, "files": results, "blocked_reason": blocked}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Include the trailing resumed files even when no new process was needed.
    summary = {"suite": args.suite, "execution": args.execution, "files": results, "blocked_reason": blocked}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.output_dir / 'summary.json'}", flush=True)
    return int(any(result["status"] not in ("passed", "compiled") for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
