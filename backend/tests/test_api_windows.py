"""Task 1: API must accept crop windows; size guard must auto-tile instead of crash."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient


def _synthetic(size=(256, 256), seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.uint8) + 120
    for _ in range(15):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)
    return img


def _client_and_capture(monkeypatch):
    from api import app as api_app

    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["out_dir"])
        stem = Path(kwargs["source_path"]).stem
        (out / f"{stem}_matches.png").write_bytes(b"png")
        return {"ok": True}

    monkeypatch.setattr(api_app, "run_pipeline", fake_run_pipeline)
    return TestClient(api_app.app), captured


def test_register_forwards_crop_windows(monkeypatch):
    """POST /api/register must parse and forward source/reference windows."""
    client, captured = _client_and_capture(monkeypatch)

    resp = client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
        },
        data={
            "sensor": "OHRC",
            "source_window": "0,1400,648,2200",
            "reference_window": "770,1768,1142,2713",
        },
    )

    assert resp.status_code == 200, resp.text
    assert captured["source_window"] == (0, 1400, 648, 2200)
    assert captured["reference_window"] == (770, 1768, 1142, 2713)


def test_register_rejects_malformed_window(monkeypatch):
    """A malformed window string must be a client error, not silently ignored."""
    client, _ = _client_and_capture(monkeypatch)

    resp = client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
        },
        data={"sensor": "OHRC", "source_window": "not-a-window"},
    )

    assert resp.status_code == 422, resp.text


def test_size_guard_auto_tiles_instead_of_crashing(tmp_path, monkeypatch, caplog):
    """An oversized image with no window must be center-tiled with a warning,
    not raise RuntimeError (Run-0 scenario)."""
    import logging

    import lunar_registration.pipeline as pl

    # Shrink the guard threshold so a 256x256 synthetic image is "oversized"
    # while the resulting tile stays large enough for PWIFT to match.
    monkeypatch.setattr(pl, "MAX_SAFE_PIXELS", 60_000, raising=False)

    src = _synthetic(seed=1)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref = cv2.warpAffine(src, M, (256, 256))
    src_path = str(tmp_path / "src.png")
    ref_path = str(tmp_path / "ref.png")
    cv2.imwrite(src_path, src)
    cv2.imwrite(ref_path, ref)

    with caplog.at_level(logging.WARNING, logger="lunar_registration.pipeline"):
        summary = pl.run_pipeline(
            source_path=src_path,
            reference_path=ref_path,
            out_dir=str(tmp_path / "out"),
            source_sensor="LROC",
            matcher="pwift",
        )

    assert "best_method" in summary
    assert "auto-tile" in caplog.text.lower() or "auto_tile" in caplog.text.lower()
