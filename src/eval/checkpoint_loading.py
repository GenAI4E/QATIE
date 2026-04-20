from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_builder import (
    build_model,
    load_checkpoint_weights,
    load_torch_checkpoint_trusted,
)

ABLATION_CHOICES = [
    "hybrid_base",
    "hybrid_ablate_fuse_refine",
    "hybrid_ablate_resolution_refine",
    "hybrid_ablate_residual",
    "hybrid_ablate_all",
]

# Variants accepted by model_builder.build_model (submission tree has no separate gate graph).
_BUILDABLE_ABLATION_ORDER = [
    "hybrid_base",
    "hybrid_ablate_fuse_refine",
    "hybrid_ablate_resolution_refine",
    "hybrid_ablate_residual",
    "hybrid_ablate_all",
]


def extract_stripped_model_state_dict(ckpt_path: str | Path) -> dict:
    """Match model_builder.load_checkpoint_weights key filtering (no load)."""
    raw = load_torch_checkpoint_trusted(ckpt_path)
    checkpoint = raw.get("state_dict", raw)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {ckpt_path!r}")

    for key in ("max_psnr_value", "max_pixel_value", "zero"):
        checkpoint.pop(key, None)

    state_dict: dict = {}
    for k, v in checkpoint.items():
        if not k.startswith("model."):
            continue
        k = k[len("model.") :]
        if k.startswith("inner_model."):
            k = k[len("inner_model.") :]
        if "weight_fake_quant" in k or "activation_post_process" in k:
            continue
        state_dict[k] = v
    return state_dict


def _ablation_from_filename(path: Path) -> str | None:
    """Infer variant from common ablation run filenames when state dicts are key-identical."""
    s = path.stem.lower()
    if "ablate_resolution_refine" in s or "resolution_refine" in s:
        return "hybrid_ablate_resolution_refine"
    if "ablate_fuse_refine" in s or "fuse_refine" in s:
        return "hybrid_ablate_fuse_refine"
    if "ablate_residual" in s:
        return "hybrid_ablate_residual"
    if "hybrid_base" in s or "component_ablation_hybrid_base" in s:
        return "hybrid_base"
    if "ablate_all" in s:
        return "hybrid_ablate_all"
    return None


def _try_hparams_channels_ablation(raw: dict) -> tuple[int | None, str | None]:
    """Read channels / ablation_model from Lightning hyper_parameters if present."""
    hp = raw.get("hyper_parameters")
    if hp is None:
        return None, None
    ch, ab = None, None
    if isinstance(hp, dict):
        ch = hp.get("channels")
        ab = hp.get("ablation_model")
    else:
        ch = getattr(hp, "channels", None)
        ab = getattr(hp, "ablation_model", None)
    if ch is not None:
        try:
            ch = int(ch)
        except (TypeError, ValueError):
            ch = None
    if ab is not None:
        ab = str(ab).lower()
    return ch, ab


def pick_ablation_and_load(
    ckpt_path: Path,
    channels: int,
    stripped_sd: dict,
    ablation_override: str | None,
) -> tuple[str, torch.nn.Module]:
    """Choose variant by minimal missing keys; load weights once."""
    if ablation_override:
        raw = ablation_override.lower()
        if raw not in ABLATION_CHOICES:
            raise ValueError(f"--ablation_model {raw!r} not in {ABLATION_CHOICES}")
        model = build_model(channels, raw)
        load_checkpoint_weights(model, str(ckpt_path))
        return raw, model

    best_name: str | None = None
    best_score: tuple[int, int] | None = None

    for name in _BUILDABLE_ABLATION_ORDER:
        model = build_model(channels, name)
        inc = model.load_state_dict(stripped_sd, strict=False)
        score = (len(inc.missing_keys), len(inc.unexpected_keys))
        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    assert best_name is not None
    model = build_model(channels, best_name)
    load_checkpoint_weights(model, str(ckpt_path))
    return best_name, model
