#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = ROOT / "third_party" / "wafer" / "build_manual"
sys.path.insert(0, str(ROOT / "scripts"))
from wafer_artifacts import read_manifest


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"ERROR: {description} not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DLCompiler Wafer wheel")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    manifest_path = require_file(build_dir / "wafer-build.json", "isolated build manifest")
    manifest = read_manifest(manifest_path)

    prebuilt_dir = build_dir / "python-package"
    prebuilt_dir.mkdir(parents=True, exist_ok=True)
    links = {prebuilt_dir / name: Path(entry["path"])
             for name, entry in manifest["artifacts"].items()}
    for destination, source in links.items():
        destination.unlink(missing_ok=True)
        destination.symlink_to(source)

    shutil.rmtree(ROOT / "build", ignore_errors=True)
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TRITON_PLUGIN_DIRS"] = str(ROOT)
    env["TRITON_VERSION"] = "3.5.0"
    env["TRITON_WHEEL_NAME"] = "triton"
    env["WAFER_PREBUILT_DIR"] = str(prebuilt_dir)
    env["WAFER_BUILD_MANIFEST"] = str(manifest_path)
    env["WAFER_LANGUAGE_DIR"] = str(ROOT / "third_party" / "wafer" / "language")
    env["WAFER_EXPERIMENTAL_DIR"] = str(ROOT / "third_party" / "wafer" / "experimental")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(args.wheel_dir.resolve()),
            str(ROOT),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    main()
