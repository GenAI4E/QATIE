from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.checkpoint_loading import (
    ABLATION_CHOICES,
    _ablation_from_filename,
    _try_hparams_channels_ablation,
    extract_stripped_model_state_dict,
    pick_ablation_and_load,
)
from src.eval.eval_tflite import load_test_data
from src.models.model_builder import (
    infer_channels_from_checkpoint,
    is_fx_quantized_full_model_checkpoint,
    load_torch_checkpoint_trusted,
)


@dataclass
class LoadedCheckpointModel:
    checkpoint_path: Path
    channels: int
    ablation_model: str
    model: torch.nn.Module


def resolve_output_paths(
    ckpt_dir: Path,
    root_arg: str | None,
    csv_arg: str | None,
    *,
    default_root_name: str,
    default_csv_name: str,
) -> tuple[Path, Path]:
    export_root = Path(root_arg).expanduser().resolve() if root_arg else ckpt_dir / default_root_name
    export_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(csv_arg).expanduser().resolve() if csv_arg else ckpt_dir / default_csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return export_root, csv_path


def validate_requested_devices(args: object) -> None:
    if getattr(args, "skip_cpu", False) and getattr(args, "skip_cuda", False):
        print("Error: both --skip_cpu and --skip_cuda", file=sys.stderr)
        raise SystemExit(1)

    if not getattr(args, "skip_cuda", False) and not torch.cuda.is_available():
        print("CUDA not available; only CPU eval will run.", file=sys.stderr)
        args.skip_cuda = True


def requested_device_names(args: object) -> list[str]:
    devices: list[str] = []
    if not getattr(args, "skip_cpu", False):
        devices.append("cpu")
    if not getattr(args, "skip_cuda", False):
        devices.append("cuda")
    return devices


def load_eval_test_data_if_needed(
    *,
    data_dir: str,
    skip_eval: bool,
    full_hd: bool,
):
    if skip_eval:
        return None

    test_data = load_test_data(data_dir, full_hd=full_hd)
    if not test_data:
        print(f"Error: no DPED test pairs under {data_dir}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Loaded {len(test_data)} test image pair(s) (full_hd={full_hd}).")
    return test_data


def load_checkpoint_model_for_export(
    ckpt_path: Path,
    *,
    ablation_override: str | None,
) -> LoadedCheckpointModel | None:
    if not ckpt_path.is_file():
        print(f"  Skip: missing or not a file (broken symlink?): {ckpt_path}", file=sys.stderr)
        return None
    if is_fx_quantized_full_model_checkpoint(ckpt_path):
        print(
            "  Skip: FX INT8 GraphModule save (from quantize.py torch.save after convert_fx). "
            "Use a Lightning checkpoint (.ckpt) with model.state_dict for ONNX->TFLite export.",
            file=sys.stderr,
        )
        return None

    try:
        raw = load_torch_checkpoint_trusted(ckpt_path)
    except Exception as exc:
        print(f"  Skip: cannot load checkpoint: {exc}", file=sys.stderr)
        return None

    hp_ch, hp_ab = _try_hparams_channels_ablation(raw if isinstance(raw, dict) else {})
    try:
        channels = int(hp_ch) if hp_ch is not None else infer_channels_from_checkpoint(str(ckpt_path))
    except Exception as exc:
        print(f"  Skip: channels: {exc}", file=sys.stderr)
        return None

    try:
        stripped = extract_stripped_model_state_dict(ckpt_path)
    except ValueError as exc:
        print(f"  Skip: {exc}", file=sys.stderr)
        return None
    if not stripped:
        print("  Skip: no model.* keys", file=sys.stderr)
        return None

    override = ablation_override
    if override is None and hp_ab and hp_ab in ABLATION_CHOICES:
        override = hp_ab
    if override is None:
        fn_ab = _ablation_from_filename(ckpt_path)
        if fn_ab and fn_ab in ABLATION_CHOICES:
            override = fn_ab

    try:
        ablation_model, model = pick_ablation_and_load(ckpt_path, channels, stripped, override)
    except Exception as exc:
        print(f"  Skip load: {exc}", file=sys.stderr)
        return None

    return LoadedCheckpointModel(
        checkpoint_path=ckpt_path,
        channels=channels,
        ablation_model=ablation_model,
        model=model,
    )
