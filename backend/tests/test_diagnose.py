"""Task 9: repair diagnose.py — the named defects and the stage-contract gap.

Named defects (plan.md Task 9):
1. `cfg.pwift_keypoint_threshold` (l.55) — that field does NOT exist on
   PipelineConfig, so every run without `--keypoint-threshold` crashed with
   AttributeError (matching.py:140 already guards it via hasattr -> 0.08).
2. `PWIFTMaps.shape` (l.25 via stats at l.92) — PWIFTMaps is a dataclass
   bundle with no .shape, so stage 2 output crashed stats() for every
   pwift_akimov sensor (OHRC/LROC).
3. Hardcoded `sensor_hint="LROC"` for the reference (l.60).
4. Single `--window` applied to BOTH images — no per-image windows.
5. Full-image OOM risk undocumented (docstring, no test).

Plus the stage-contract gap: stage 2 returns PWIFTMaps for pwift_akimov
sensors but stages 3a-3c called the generic path unconditionally — diagnose
must dispatch exactly like matching.py:259 (BOTH PWIFTMaps -> paper path;
else generic on the M_PW map extracted from any PWIFTMaps side).
"""
import sys

import numpy as np
import pytest


def _synthetic_pair(tmp_path, size=256, seed=11):
    import cv2

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8) + 120
    for _ in range(15):
        cx = int(rng.integers(20, size - 20))
        cy = int(rng.integers(20, size - 20))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref = cv2.warpAffine(img, M, (size, size))
    src_path = str(tmp_path / "dg_src.tif")
    ref_path = str(tmp_path / "dg_ref.tif")
    cv2.imwrite(src_path, img)
    cv2.imwrite(ref_path, ref)
    return src_path, ref_path


def test_stats_handles_pwift_maps_bundle(capsys):
    """Defect 2: stats() on a PWIFTMaps bundle must not read .shape."""
    import diagnose
    from lunar_registration.pwift import PWIFTMaps

    maps = PWIFTMaps(
        M_PW=np.random.default_rng(0).random((32, 32)).astype(np.float32),
        m_PW=np.zeros((32, 32), np.float32),
        MIM=np.zeros((32, 32), np.int32),
        w=np.ones((32, 32), np.float32),
        w_soft=np.ones((32, 32), np.float32),
        mask=np.ones((32, 32), bool),
    )
    diagnose.stats("src_illum", maps)  # AttributeError .shape before Task 9
    out = capsys.readouterr().out
    assert "PWIFTMaps" in out
    assert "M_PW" in out


def test_diagnose_runs_without_keypoint_threshold_field(tmp_path, monkeypatch, capsys):
    """Defect 1: a run that does NOT pass --keypoint-threshold used to read
    the nonexistent cfg.pwift_keypoint_threshold at l.55 -> AttributeError.
    Also covers the paper-path dispatch (LROC/LROC -> both PWIFTMaps)."""
    import diagnose

    src_path, ref_path = _synthetic_pair(tmp_path)
    # No --keypoint-threshold: this is the exact shape of the l.55 crash
    # (reading cfg.pwift_keypoint_threshold that PipelineConfig never had).
    # Reference sensor omitted on purpose: autodetect must resolve it so
    # BOTH sides are pwift_akimov (PWIFTMaps -> paper path dispatch).
    monkeypatch.setattr(sys, "argv", [
        "diagnose.py", "--source", src_path, "--reference", ref_path,
        "--source-sensor", "LROC",
        "--window", "0,0,256,256",
    ])
    diagnose.main()
    out = capsys.readouterr().out
    # the knobs line prints fields that actually exist
    assert "pwift_ratio_test" in out
    assert "score_percentile" in out
    # dispatch: both PWIFTMaps -> paper path, stages ran
    assert "illumination path: paper" in out
    assert "n_keypoints" in out
    assert "n_matches" in out
    # defect 2: stage-2 bundle went through stats() without crashing
    assert "PWIFTMaps[" in out


def test_diagnose_reference_sensor_and_per_image_windows(tmp_path, monkeypatch, capsys):
    """Defect 3+4: reference sensor flag (was hardcoded LROC) and separate
    --source-window/--reference-window (each image cropped to its own size)."""
    import diagnose

    src_path, ref_path = _synthetic_pair(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "diagnose.py", "--source", src_path, "--reference", ref_path,
        "--source-sensor", "LROC", "--reference-sensor", "TMC",
        "--source-window", "0,0,256,256",
        "--reference-window", "8,8,240,240",
    ])
    diagnose.main()
    out = capsys.readouterr().out

    assert "ref sensor: TMC" in out          # defect 3: not hardcoded LROC
    assert "shape=(256, 256)" in out         # source window applied
    assert "shape=(240, 240)" in out         # reference window applied independently
    # mixed representations -> generic path via M_PW extraction (matching.py:259)
    assert "illumination path: generic" in out
    assert "n_matches" in out
