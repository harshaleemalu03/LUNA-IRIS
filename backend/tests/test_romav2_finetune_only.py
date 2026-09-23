"""Task 5 (scope lock 2): RoMa must load ONLY the fine-tuned checkpoint.

Covers the whole contract in one construction:
- `RoMaV2.__init__` accepts caller weights (parameterizes the vendored ctor)
- `matching.py` passes ROMA2 fine-tune through — single strict load
- no GitHub fetch of base romav2.0.1.pt happens on the pipeline's cold start
- §4.8 key-set identity: fine-tune keys ≡ architecture keys (strict-able)
"""

import pytest
import torch

from lunar_registration.config import PipelineConfig
from lunar_registration.matching import resolve_romav2_weights


def test_finetune_only_load_no_base_download(monkeypatch):
    weights_path = resolve_romav2_weights()
    assert weights_path.exists(), f"fine-tune checkpoint missing: {weights_path}"

    def _no_network(*args, **kwargs):
        raise AssertionError(
            "Task 5 violation: pipeline attempted to fetch the base "
            "romav2.0.1.pt from GitHub — fine-tune-only is mandated"
        )

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _no_network)

    # Constructs RoMaV2 internally and must succeed WITHOUT the hub call:
    # proves the constructor now takes caller weights (RED before Task 5:
    # AssertionError from _no_network, raised inside RoMaV2.__init__).
    from lunar_registration.matching import get_matcher

    matcher = get_matcher("roma2", PipelineConfig())

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    fine_sd = ckpt.get("model", ckpt)

    # §4.8: key sets identical in both directions -> strict load is sound.
    model_sd = matcher.model.state_dict()
    assert set(model_sd) == set(fine_sd)
    assert not (set(model_sd) - set(fine_sd))
    assert not (set(fine_sd) - set(model_sd))

    # The loaded weights ARE the fine-tune (the base checkpoint was never
    # fetched, so any equality here can only come from ROMA2_WEIGHTS_PATH).
    key = next(iter(fine_sd))
    assert torch.allclose(
        model_sd[key].float().cpu(),
        fine_sd[key].float().cpu(),
        atol=0,
        rtol=0,
    )

    # Explicit strict=True re-assert (the §4.8 claim as a living test).
    matcher.model.load_state_dict(fine_sd, strict=True)
