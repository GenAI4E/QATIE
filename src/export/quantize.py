"""
Convert trained checkpoints to INT8 using the same FX QAT pipeline as train_qat.py.

- QAT Lightning checkpoints: load fake-quant observer state from the checkpoint — no image
  calibration required.
- FP32 checkpoints: load conv weights, run observer warmup (calibration) over data unless
  you pass --calibrated_qat_path from a previous run.
"""
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.ao.quantization.quantize_fx as quantize_fx

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dped_dataset import DPEDDataModule
from src.eval.eval_pytorch import evaluate_model
from src.models.model_builder import (
    build_model,
    build_qat_qconfig_mapping,
    infer_channels_from_checkpoint,
    load_checkpoint_weights,
    load_torch_checkpoint_trusted,
)


def prepare_qat_model(model, device, legacy_qat_graph: bool = False):
    """FX Graph Mode QAT — must match train_qat.prepare_qat_model when legacy_qat_graph is False."""
    backend = "qnnpack"
    torch.backends.quantized.engine = backend

    qconfig_mapping = build_qat_qconfig_mapping(concat_fp32_in_fp32=not legacy_qat_graph)
    example_inputs = (torch.randn(1, 3, 100, 100),)

    model.train()
    model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    model.to(device)
    return model


def convert_qat_model(model):
    """Convert a QAT-prepared FX model to a fully quantized INT8 model."""
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        model = quantize_fx.convert_fx(model)
    return model


def _strip_lightning_prefix(state_dict):
    """Lightning saves keys as model.*; optional inner_model.* from older wrappers."""
    out = {}
    for k, v in state_dict.items():
        if not k.startswith("model."):
            continue
        k = k[len("model.") :]
        if k.startswith("inner_model."):
            k = k[len("inner_model.") :]
        out[k] = v
    return out


def _state_dict_has_fx_qat_observers(state_dict):
    for k in state_dict:
        if "weight_fake_quant" in k or "activation_post_process" in k:
            return True
    return False


def _load_checkpoint_raw(ckpt_path):
    return load_torch_checkpoint_trusted(ckpt_path)


def load_into_prepared_qat(qat_model, ckpt_path, legacy_qat_graph: bool = False):
    """Load Lightning checkpoint weights into an already FX-prepared QAT model."""
    ckpt = _load_checkpoint_raw(ckpt_path)
    sd = ckpt.get("state_dict", ckpt)
    stripped = _strip_lightning_prefix(sd)
    incompatible = qat_model.load_state_dict(stripped, strict=False)
    if incompatible.missing_keys:
        n = min(8, len(incompatible.missing_keys))
        print(
            f"[load_into_prepared_qat] missing {len(incompatible.missing_keys)} key(s) "
            f"(showing {n}): {incompatible.missing_keys[:n]}"
        )
    if incompatible.unexpected_keys:
        n = min(8, len(incompatible.unexpected_keys))
        print(
            f"[load_into_prepared_qat] unexpected {len(incompatible.unexpected_keys)} key(s) "
            f"(showing {n}): {incompatible.unexpected_keys[:n]}"
        )
    if (incompatible.missing_keys or incompatible.unexpected_keys) and not legacy_qat_graph:
        print(
            "[load_into_prepared_qat] Hint: if this checkpoint was trained with the older FX QAT graph "
            "(ConcatFP32 quantized, not FP32), re-run with --legacy_qat_graph."
        )
    return incompatible


def convert_checkpoint_to_int8_model(
    ckpt_path: str,
    *,
    ablation_model: str = "hybrid_base",
    channels: int | None = None,
    legacy_qat_graph: bool = False,
    device: torch.device | None = None,
    data_dir: str | None = None,
    calibrated_qat_path: str | None = None,
    save_path: str | None = None,
    full_hd: bool = False,
) -> torch.nn.Module:
    """
    Load a Lightning checkpoint and return the converted INT8 FX model (same pipeline as __main__).

    Use this when ``torch.load`` on a saved ``*_int8.pth`` fails (FX GraphModule pickle can break
    across PyTorch versions). Does not write files unless ``save_path`` is set (then saves
    calibrated QAT dict / int8 .pth like the CLI).
    """
    if device is None:
        device = torch.device("cpu")

    ch = channels
    if ch is None:
        try:
            ch = infer_channels_from_checkpoint(ckpt_path)
        except Exception:
            ch = 24

    fp32_model = build_model(ch, model_name=ablation_model).to(device)

    ckpt = _load_checkpoint_raw(ckpt_path)
    raw_sd = ckpt.get("state_dict", ckpt)
    has_fx_qat = _state_dict_has_fx_qat_observers(raw_sd)

    if has_fx_qat:
        qat_model = prepare_qat_model(fp32_model, device, legacy_qat_graph=legacy_qat_graph)
        load_into_prepared_qat(qat_model, ckpt_path, legacy_qat_graph=legacy_qat_graph)
        needs_warmup = False
    else:
        load_checkpoint_weights(fp32_model, ckpt_path)
        qat_model = prepare_qat_model(fp32_model, device, legacy_qat_graph=legacy_qat_graph)
        needs_warmup = True

    if calibrated_qat_path is not None:
        qat_model.load_state_dict(torch.load(calibrated_qat_path, map_location=device))
        qat_model.to(device)
        qat_model.eval()
        needs_warmup = False
    elif needs_warmup:
        if data_dir is None:
            print(
                "Warning: FP32→INT8 without observer warmup — scales may be default. "
                "Pass data_dir for warmup or calibrated_qat_path from a prior run."
            )
        else:
            run_observer_warmup(qat_model, data_dir, device, full_hd=full_hd)
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                tag = f"{ablation_model}_c{ch}"
                calibrated_path = os.path.join(save_path, f"{tag}_calibrated_qat.pth")
                torch.save(qat_model.state_dict(), calibrated_path)
                print(f"Saved QAT state dict to {calibrated_path}")

    qat_model.eval()
    try:
        model_int8 = convert_qat_model(qat_model)
    except Exception as e:
        print(f"Error during convert_fx: {e}")
        raise

    dummy_input = torch.randn(1, 3, 100, 100)
    with torch.no_grad():
        out = model_int8(dummy_input)
    print(f"INT8 sanity check output shape: {tuple(out.shape)}")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        tag = f"{ablation_model}_c{ch}"
        int8_path = os.path.join(save_path, f"{tag}_int8.pth")
        torch.save(model_int8, int8_path)
        print(f"Saved INT8 model to {int8_path}")

    return model_int8


def run_observer_warmup(qat_model, data_dir, device, batch_size=4, num_workers=4, full_hd: bool=False):
    """Populate FX fake-quant observers before convert_fx (FP32 → QAT path only)."""
    from tqdm import tqdm

    data_path_list = [("iphone", Path(data_dir) / "iphone")]
    cal_dm = DPEDDataModule(data_dir=data_path_list, patch_data=not full_hd, batch_size=batch_size, num_workers=num_workers)
    cal_dm.setup(stage="test")
    qat_model.to(device)
    qat_model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(cal_dm.test_dataloader(), desc="Observer warmup"):
            qat_model(inputs.to(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HybridMixUNet checkpoint to INT8 (FX QAT, same as train_qat.py)")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to Lightning checkpoint (.ckpt) — FP32 or QAT-trained",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved INT8 .pth model to load and evaluate directly (skips build/convert)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/tinhanh/MobileAI/mixedmodel/ckpt/int8",
        help="Directory to save the converted INT8 model and optional calibrated QAT dict",
    )
    parser.add_argument(
        "--calibrated_qat_path",
        type=str,
        default=None,
        help="Path to a saved QAT state dict (.pth) with observers — skips observer warmup",
    )
    parser.add_argument(
        "--full_hd",
        action="store_true",
        help="Use full-hd test data",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root for observer warmup (FP32→QAT) and/or evaluation (iphone/ subfolder, full-hd test data)",
    )
    parser.add_argument("--channels", type=int, default=None, help="Base channel width c (default: infer from ckpt or 24)")
    parser.add_argument(
        "--ablation_model",
        type=str,
        default="hybrid_base",
        choices=[
            "hybrid_base",
            "hybrid_ablate_fuse_refine",
            "hybrid_ablate_resolution_refine",
            "hybrid_ablate_residual",
            "hybrid_ablate_all",
        ],
        help="Must match train_qat.py / checkpoint architecture",
    )
    parser.add_argument("--metrics_file", type=str, default=None, help="Optional JSON path for eval metrics")
    parser.add_argument(
        "--legacy_qat_graph",
        action="store_true",
        help=(
            "Build the same FX QAT graph as older train_qat (ConcatFP32 not kept in FP32). "
            "Use this so existing QAT Lightning checkpoints load with strict key alignment; "
            "omit for new checkpoints trained with current train_qat.py."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    args = parser.parse_args()

    device = torch.device(args.device)
    channels_out = None

    if args.model_path is not None:
        print(f"Loading INT8 model from {args.model_path}...")
        model_int8 = torch.load(args.model_path, map_location=args.device)
        print("Model loaded successfully.")
    else:
        if not args.ckpt_path:
            raise SystemExit("Provide --ckpt_path (or --model_path for pre-exported INT8).")

        channels = args.channels
        if channels is None:
            try:
                channels = infer_channels_from_checkpoint(args.ckpt_path)
            except Exception as e:
                print(f"Could not infer channels ({e}); using default 24.")
                channels = 24

        channels_out = channels

        ckpt = _load_checkpoint_raw(args.ckpt_path)
        raw_sd = ckpt.get("state_dict", ckpt)
        has_fx_qat = _state_dict_has_fx_qat_observers(raw_sd)

        if has_fx_qat and args.legacy_qat_graph:
            print(
                "Using legacy QAT graph (matches pre–ConcatFP32-FP32 training) — "
                "checkpoint keys should align."
            )
        if has_fx_qat:
            print("Checkpoint contains FX QAT fake-quant tensors — preparing QAT graph and loading weights.")
        else:
            print("Checkpoint looks FP32 — loading conv weights, then inserting FX QAT (observers need warmup).")

        print(f"Building {args.ablation_model} with channels={channels}...")
        if args.calibrated_qat_path is not None:
            print(f"Loading QAT state dict from {args.calibrated_qat_path}...")
            print("Loaded saved QAT state; skipping observer warmup.")

        model_int8 = convert_checkpoint_to_int8_model(
            args.ckpt_path,
            ablation_model=args.ablation_model,
            channels=channels,
            legacy_qat_graph=args.legacy_qat_graph,
            device=device,
            data_dir=args.data_dir,
            calibrated_qat_path=args.calibrated_qat_path,
            save_path=args.save_path,
            full_hd=args.full_hd,
        )

    if args.data_dir is not None:
        psnr, ssim = evaluate_model(model_int8, args.data_dir, device=device, full_hd=args.full_hd)
        if args.metrics_file:
            metrics_path = Path(args.metrics_file)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ablation_model": getattr(args, "ablation_model", None),
                "channels": channels_out,
                "psnr": psnr,
                "ssim": ssim,
            }
            metrics_path.write_text(json.dumps(payload, indent=2))
            print(f"Saved eval metrics to {metrics_path}")
    else:
        print("No --data_dir provided, skipping evaluation.")
