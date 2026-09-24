"""Task 10: resample_to_gsd was dead code (zero references repo-wide).

WHY these tests assert ABSENCE: the plan's choice was "wire or delete" —
deleting is the ruling (wiring would pre-transform every matcher's inputs,
an algorithmic change outside the scope locks, and would risk double-
applying scale on top of the Task-2 GSD prior band). These tests guard
the deletion: the symbol must not exist, and no package source may
reference it (a dangling call site would NameError at runtime anyway,
but a grep-visible guard keeps the dead name from creeping back).
"""
import pathlib

import lunar_registration.preprocessing as preprocessing

PACKAGE_DIR = pathlib.Path(preprocessing.__file__).parent


def test_resample_to_gsd_symbol_removed():
    """resample_to_gsd must not exist — it was deleted as dead code."""
    assert not hasattr(preprocessing, "resample_to_gsd")


def test_no_package_source_references_resample_to_gsd():
    """No file under lunar_registration/ may mention the dead name."""
    offenders = [
        str(p.relative_to(PACKAGE_DIR))
        for p in sorted(PACKAGE_DIR.rglob("*.py"))
        if "resample_to_gsd" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == [], f"dead name still referenced in: {offenders}"
