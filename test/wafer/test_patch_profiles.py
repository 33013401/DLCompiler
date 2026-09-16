"""Check profile composition against the actual pinned Triton Git objects."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("apply_triton_profile", ROOT / "scripts/apply_triton_profile.py")
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


@pytest.mark.parametrize("profile", ["ascend", "wafer-tools", "wafer-frontend"])
def test_profiles_apply_idempotently_and_preserve_edits(tmp_path, profile):
    source = tmp_path / "triton"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(ROOT / "third_party/triton"), str(source)],
                   check=True, capture_output=True)
    commit = "c3c476f357f1e9768ea4e45aa5c17528449ab9ef"
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], check=True, capture_output=True)
    first = profiles.apply_profile(source, profile)
    diff = profiles.git(source, "diff", "--binary")
    assert profiles.apply_profile(source, profile) == first
    assert profiles.apply_profile(source, profile, check=True) == first
    assert profiles.git(source, "diff", "--binary") == diff
    edited = source / "python/src/ir.cc"
    edited.write_text(edited.read_text() + "\n// independent user change\n")
    with pytest.raises(RuntimeError, match="Refusing to replace modified"):
        profiles.apply_profile(source, profile)
    assert edited.read_text().endswith("// independent user change\n")
