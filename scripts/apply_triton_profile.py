#!/usr/bin/env python3
"""Apply one pinned patch profile without resetting or staging caller changes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "patch/triton/profiles/profiles.json"


def git(source, *args, env=None, data=None):
    return subprocess.check_output(
        ["git", "-C", str(source), *args], env=env, input=data, stderr=subprocess.PIPE
    )


def apply_profile(source, profile, check=False):
    source = Path(source).resolve()
    catalog = json.loads(CATALOG.read_text())
    base = catalog["triton_commit"]
    if git(source, "rev-parse", "HEAD").decode().strip() != base:
        raise RuntimeError(f"{profile} requires Triton {base}: {source}")
    patches = [ROOT / "patch/triton" / p for p in catalog["profiles"][profile]]
    identity = hashlib.sha256()
    for path in patches:
        identity.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        identity.update(path.read_bytes())

    # Build the expected result in a temporary Git index. Overlapping patches
    # are applied in order; the user's index and working files are untouched.
    with tempfile.TemporaryDirectory(prefix="triton-profile-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git(source, "read-tree", base, env=env)
        for path in patches:
            git(source, "apply", "--cached", "--whitespace=nowarn", str(path), env=env)
        tree = git(source, "write-tree", env=env).decode().strip()
        paths = git(source, "diff", "--name-only", base, tree).decode().splitlines()
        expected = {p: git(source, "show", f"{tree}:{p}") for p in paths}
        dirty = set(git(source, "diff", "--name-only", "HEAD").decode().splitlines())
        matches = dirty <= set(paths) and all(
            (source / p).is_file() and (source / p).read_bytes() == data
            for p, data in expected.items()
        )
        if not matches:
            if check:
                raise RuntimeError(f"Source does not match {profile}: {source}")
            if dirty:
                raise RuntimeError(
                    f"Refusing to replace modified Triton source at {source}. "
                    "Use a clean checkout for this profile; no reset was performed."
                )
            delta = git(source, "diff", "--binary", base, tree)
            git(source, "apply", "--check", "-", data=delta)
            git(source, "apply", "-", data=delta)
    return {"profile": profile, "triton_commit": base, "patch_sha256": identity.hexdigest(),
            "source": str(source), "files": {
                p: hashlib.sha256(data).hexdigest() for p, data in expected.items()
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/triton")
    parser.add_argument("--profile", choices=json.loads(CATALOG.read_text())["profiles"], required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    try:
        result = apply_profile(args.source, args.profile, args.check)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.decode(errors="replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise SystemExit(detail) from None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Triton profile verified: {args.profile} ({result['patch_sha256'][:12]})")


if __name__ == "__main__":
    main()
