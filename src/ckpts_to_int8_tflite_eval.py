"""
For every Lightning checkpoint under a directory: export ONNX → TF SavedModel → **INT8 PTQ TFLite**
(TensorFlow Lite post-training quantization with a DPED representative dataset), then evaluate
PSNR/SSIM with eval_pytorch.evaluate_model() on CPU and CUDA (same as ckpts_to_tflite_eval.py).

TFLite runs on the TF Lite interpreter (CPU). The CUDA pass keeps tensors on GPU around the
wrapper so evaluate_model()'s device placement matches a PyTorch workflow; numerically the
TFLite path is the same as CPU.

Dynamic shape + PTQ may fail on some TensorFlow versions; errors are printed per checkpoint.

Run from submission/Source-Codes:
  python ckpts_to_int8_tflite_eval.py --ckpt_dir ./ckpts --data_dir /path/to/dped --output_csv ./ckpts/tflite_int8_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_tflite_utils import (
    load_checkpoint_model_for_export,
    load_eval_test_data_if_needed,
    requested_device_names,
    resolve_output_paths,
    validate_requested_devices,
)
from src.data.dped_dataset import DPEDDataModule
from src.eval.eval_tflite import evaluate_tflite
from src.export.to_tflite import convert_pytorch_to_int8_tflite

from src.eval.benchmark_ckpts import (
    _gather_checkpoint_paths,
)

class TFLiteTorchModule(nn.Module):
    """Wraps a TFLite model so evaluate_tflite() can call forward() like PyTorch."""

    def __init__(self, tflite_path: str | Path):
        super().__init__()
        self.tflite_path = str(tflite_path)
        import tensorflow as tf

        self._tf = tf
        # Disable XNNPACK delegate for robustness: some INT8 PTQ graphs cannot
        # be reshaped/prepared by the delegate after tensor resizing.
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_path, experimental_delegates=[])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self._expects_nchw: bool | None = None

    def _infer_layout(self) -> bool:
        """True if TFLite tensor is NCHW, False if NHWC."""
        shape = list(self.input_details[0]["shape"])
        if len(shape) != 4:
            raise ValueError(f"Expected 4D input, got shape {shape}")
        return shape[1] == 3 and shape[-1] != 3

    def _prepare_input(self, x_np: np.ndarray) -> np.ndarray:
        if self._expects_nchw is None:
            self._expects_nchw = self._infer_layout()
        if self._expects_nchw:
            return x_np.astype(np.float32)
        return np.transpose(x_np, (0, 2, 3, 1)).astype(np.float32)

    @staticmethod
    def _to_nchw(out: np.ndarray) -> np.ndarray:
        if out.ndim != 4:
            raise ValueError(f"Expected 4D output, got {out.shape}")
        if out.shape[1] == 3:
            return out
        if out.shape[-1] == 3:
            return np.transpose(out, (0, 3, 1, 2))
        raise ValueError(f"Cannot infer CHW layout from shape {out.shape}")

    @staticmethod
    def _quantize_if_needed(arr: np.ndarray, details: dict) -> np.ndarray:
        dtype = details["dtype"]
        if not np.issubdtype(dtype, np.integer):
            return arr.astype(dtype, copy=False)
        scale, zero = details.get("quantization", (0.0, 0))
        scale = float(scale)
        zero = int(zero)
        if scale <= 0:
            raise ValueError(f"Invalid input quantization scale: {scale}")
        q = np.round(arr / scale + zero)
        qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
        return np.clip(q, qmin, qmax).astype(dtype)

    @staticmethod
    def _dequantize_if_needed(arr: np.ndarray, details: dict) -> np.ndarray:
        dtype = details["dtype"]
        if not np.issubdtype(dtype, np.integer):
            return arr.astype(np.float32, copy=False)
        scale, zero = details.get("quantization", (0.0, 0))
        scale = float(scale)
        zero = int(zero)
        if scale <= 0:
            raise ValueError(f"Invalid output quantization scale: {scale}")
        return (arr.astype(np.float32) - zero) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
        interpreter = self.interpreter
        input_details = self.input_details
        output_details = self.output_details

        # Proactively resize inputs to the model's expected (static) HxW.
        # This avoids re-allocations that can break with INT8 PTQ + delegates.
        if self._expects_nchw is None:
            self._expects_nchw = self._infer_layout()
        exp_shape = list(input_details[0]["shape"])
        if self._expects_nchw:
            exp_h, exp_w = exp_shape[2], exp_shape[3]
        else:
            exp_h, exp_w = exp_shape[1], exp_shape[2]
        if exp_h is not None and exp_w is not None and exp_h > 0 and exp_w > 0:
            if x.shape[-2] != exp_h or x.shape[-1] != exp_w:
                x = F.interpolate(x, size=(exp_h, exp_w), mode="bilinear", align_corners=False)

        x_np = x.detach().cpu().numpy()
        inp = self._prepare_input(x_np)

        if list(inp.shape) != list(input_details[0]["shape"]):
            interpreter.resize_tensor_input(input_details[0]["index"], list(inp.shape))
            interpreter.allocate_tensors()
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            input_details = self.input_details
            output_details = self.output_details

        inp_prepared = self._quantize_if_needed(inp, input_details[0])
        interpreter.set_tensor(input_details[0]["index"], inp_prepared)
        interpreter.invoke()
        out_raw = interpreter.get_tensor(output_details[0]["index"])
        out = self._dequantize_if_needed(out_raw, output_details[0])
        out = self._to_nchw(out)
        out_t = torch.from_numpy(out)
        if out_t.shape[-2] != orig_h or out_t.shape[-1] != orig_w:
            out_t = F.interpolate(out_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return out_t.to(dev)

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def make_dped_representative_dataset(
    data_dir: str,
    input_h: int,
    input_w: int,
    max_samples: int,
    *,
    full_hd: bool = False,
):
    """Build a ``representative_dataset`` callable for TFLiteConverter (NHWC float32, fixed H×W).

    Uses the same ``DPEDDataModule`` layout as ``eval_pytorch.evaluate_model``:
    ``patch_data=not full_hd`` (patches under ``test_data/patches/`` vs full-res under
    ``<phone>/*.jpg`` / ``canon/*.jpg`` — see ``dped_dataset.DPEDData``).
    """

    def representative_dataset():
        data_path_list = [(phone, Path(data_dir) / phone) for phone in os.listdir(data_dir)]
        dm = DPEDDataModule(
            data_dir=data_path_list,
            patch_data=not full_hd,
            batch_size=1,
            num_workers=0,
        )
        dm.setup(stage="test")
        loader = dm.test_dataloader()
        n = 0
        for inputs, _ in loader:
            x = inputs
            if x.shape[2] != input_h or x.shape[3] != input_w:
                x = F.interpolate(x, size=(input_h, input_w), mode="bilinear", align_corners=False)
            x_np = x[0].permute(1, 2, 0).numpy().astype(np.float32)
            yield [np.expand_dims(x_np, 0)]
            n += 1
            if n >= max_samples:
                break

    return representative_dataset


def parse_args() -> argparse.Namespace:
    default_ckpt_dir = _script_dir() / "ckpts"
    p = argparse.ArgumentParser(
        description="Convert ckpts to INT8 PTQ TFLite and evaluate with eval_tflite.evaluate_tflite."
    )
    p.add_argument("--ckpt_dir", type=str, default=str(default_ckpt_dir), help="Directory with .ckpt / .pt files")
    p.add_argument(
        "--full_hd",
        action="store_true",
        help="Use full-resolution test images (DPEDData patch_data=False) for eval and PTQ calibration",
    )
    p.add_argument("--data_dir", type=str, required=True, help="DPED root (iphone/ …) for calibration + test eval")
    p.add_argument(
        "--tflite_root",
        type=str,
        default=None,
        help="Directory for exported TFLite trees (default: <ckpt_dir>/tflite_int8_exports)",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="CSV path (default: <ckpt_dir>/tflite_int8_eval_cpu_cuda.csv)",
    )
    p.add_argument("--input_h", type=int, default=100, help="ONNX export and calibration height")
    p.add_argument("--input_w", type=int, default=100, help="ONNX export and calibration width")
    p.add_argument("--dynamic", action="store_true", help="Dynamic H,W (uses model_none_int8_ptq.tflite)")
    p.add_argument(
        "--calibration_samples",
        type=int,
        default=128,
        help="Number of DPED test samples for PTQ calibration (patches or full-res, see --full_hd)",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_cuda", action="store_true", help="Do not emit CSV rows with device=cuda (metrics are TFLite-on-CPU)")
    p.add_argument("--skip_cpu", action="store_true", help="Do not emit CSV rows with device=cpu")
    p.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    p.add_argument("--ablation_model", type=str, default=None, help="Force variant (see train_qat / model_builder)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset for export (default 18)",
    )
    p.add_argument(
        "--legacy-onnx",
        dest="legacy_onnx",
        action="store_true",
        default=False,
        help="TorchScript ONNX export (dynamo=False); use with --opset 11 if needed",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.is_dir():
        print(f"Error: not a directory: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    tflite_root, out_path = resolve_output_paths(
        ckpt_dir,
        args.tflite_root,
        args.output_csv,
        default_root_name="tflite_int8_exports",
        default_csv_name="tflite_int8_eval_cpu_cuda.csv",
    )

    if args.calibration_samples < 1:
        print("Error: --calibration_samples must be >= 1", file=sys.stderr)
        sys.exit(1)

    paths = _gather_checkpoint_paths(ckpt_dir)
    if not paths:
        print(f"No .ckpt or .pt files under {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    validate_requested_devices(args)
    test_data = load_eval_test_data_if_needed(
        data_dir=args.data_dir,
        skip_eval=args.skip_eval,
        full_hd=args.full_hd,
    )
    print(f"Skip eval? {args.skip_eval}")
    fieldnames = [
        "checkpoint_path",
        "tflite_path",
        "ablation_model",
        "channels",
        "device",
        "quantization",
        "calibration_samples",
        "export_sec",
        "eval_sec",
        "psnr",
        "ssim",
    ]
    rows: list[dict] = []

    t_run0 = time.perf_counter()
    for ckpt_path in paths:
        print(f"\n=== {ckpt_path} ===")
        loaded = load_checkpoint_model_for_export(
            ckpt_path,
            ablation_override=args.ablation_model,
        )
        if loaded is None:
            continue

        rep_dataset = make_dped_representative_dataset(
            args.data_dir,
            args.input_h,
            args.input_w,
            args.calibration_samples,
            full_hd=args.full_hd,
        )

        stem = loaded.checkpoint_path.stem
        export_dir = tflite_root / stem
        try:
            t_export0 = time.perf_counter()
            tflite_path = convert_pytorch_to_int8_tflite(
                loaded.model,
                export_dir,
                stem,
                args.input_h,
                args.input_w,
                args.dynamic,
                rep_dataset,
                opset_version=args.opset,
                legacy_onnx=args.legacy_onnx,
            )
            export_sec = time.perf_counter() - t_export0
        except Exception as e:
            print(f"  INT8 TFLite export failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

        print(f"  INT8 PTQ TFLite: {tflite_path}  (export {export_sec:.2f}s)")

        devices = requested_device_names(args)

        for dev_name in devices:
            t_eval0 = time.perf_counter()
            if not args.skip_eval:
                psnr, ssim = evaluate_tflite(
                    tflite_path,
                    test_data,
                    use_xnnpack=False,
                    eval_height=args.input_h,
                    eval_width=args.input_w,
                )
            else:
                psnr, ssim = 0.0, 0.0
            eval_sec = time.perf_counter() - t_eval0
            print(f"  [{dev_name}] {eval_sec:.2f}s  PSNR={psnr:.4f}  SSIM={ssim:.4f}")
            rows.append(
                {
                    "checkpoint_path": str(loaded.checkpoint_path),
                    "tflite_path": str(tflite_path),
                    "ablation_model": loaded.ablation_model,
                    "channels": loaded.channels,
                    "device": dev_name,
                    "quantization": "ptq_int8",
                    "calibration_samples": str(args.calibration_samples),
                    "export_sec": f"{export_sec:.3f}",
                    "eval_sec": f"{eval_sec:.3f}",
                    "psnr": f"{psnr:.6f}",
                    "ssim": f"{ssim:.6f}",
                }
            )

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total_sec = time.perf_counter() - t_run0
    print(f"\nWrote {len(rows)} row(s) to {out_path}")
    print(f"Total runtime: {total_sec:.2f}s ({total_sec / 60.0:.2f} min)")


if __name__ == "__main__":
    main()
