from pathlib import Path
import numpy as np
import pytest

from lunar_registration.config import PipelineConfig
from lunar_registration.matching import (
    BaseMatcher,
    EloftrMatcher,
    MissingWeights,
    PWIFTMatcher,
    Roma2Matcher,
    get_matcher,
    resolve_romav2_weights,
    resolve_eloftr_checkpoint,
)


def test_matcher_factory_pwift():
    matcher = get_matcher("pwift")
    assert isinstance(matcher, BaseMatcher)
    assert isinstance(matcher, PWIFTMatcher)


def test_missing_weights_fails_loudly():
    with pytest.raises(MissingWeights):
        Roma2Matcher(weights_path="/path/to/nonexistent/weights.pt")

    with pytest.raises(MissingWeights):
        EloftrMatcher(checkpoint_path="/path/to/nonexistent/ckpt.ckpt")


def test_pwift_matcher_synthetic():
    cfg = PipelineConfig()
    matcher = get_matcher("pwift", cfg)
    im0 = np.ones((128, 128), dtype=np.float32) * 0.5
    im1 = np.ones((128, 128), dtype=np.float32) * 0.5
    res = matcher.match(im0, im1)
    assert res.method == "pwift"
    assert hasattr(res, "pts_src")
    assert hasattr(res, "pts_dst")


def test_roma2_matcher_initialization():
    weights_path = resolve_romav2_weights()
    if not weights_path.exists():
        pytest.skip("RoMa v2 fine-tuned weights not found")

    matcher = Roma2Matcher(weights_path=weights_path, device="cpu")
    assert isinstance(matcher, BaseMatcher)
    assert matcher.model is not None


def test_eloftr_matcher_initialization():
    ckpt_path = resolve_eloftr_checkpoint()
    if not ckpt_path.exists():
        pytest.skip("ELoFTR fine-tuned checkpoint not found")

    matcher = EloftrMatcher(checkpoint_path=ckpt_path, device="cpu")
    assert isinstance(matcher, BaseMatcher)
    assert matcher.model is not None


def test_roma2_tile_iou_pairing():
    """Verify that tile pairing with IoU >= 0.40 keeps co-located pairs and discards 85% disjoint pairs."""
    cfg = PipelineConfig(roma2_min_tile_iou=0.40)
    # Mock matcher instance with _generate_tiles
    weights_path = resolve_romav2_weights()
    if not weights_path.exists():
        pytest.skip("RoMa v2 fine-tuned weights not found")

    matcher = Roma2Matcher(weights_path=weights_path, device="cpu", cfg=cfg)
    h, w = 6600, 600
    img = np.zeros((h, w), dtype=np.uint8)
    tiles = matcher._generate_tiles(img, tile_size=800, overlap=0.15)
    assert len(tiles) == 10

    # Build pairs as in matcher.match()
    tile_pairs = []
    min_iou = cfg.roma2_min_tile_iou
    for t_s in tiles:
        bb_s = t_s["bbox"]
        area_s = bb_s[2] * bb_s[3]
        for t_r in tiles:
            bb_r = t_r["bbox"]
            area_r = bb_r[2] * bb_r[3]
            ix0 = max(bb_s[0], bb_r[0])
            iy0 = max(bb_s[1], bb_r[1])
            ix1 = min(bb_s[0] + bb_s[2], bb_r[0] + bb_r[2])
            iy1 = min(bb_s[1] + bb_s[3], bb_r[1] + bb_r[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = area_s + area_r - inter
            iou = inter / max(1, union)
            if iou >= min_iou:
                tile_pairs.append((t_s, t_r))

    # All pairs must be diagonal co-located (tile i with tile i)
    assert len(tile_pairs) == 10
    for ts, tr in tile_pairs:
        assert ts["bbox"] == tr["bbox"]

