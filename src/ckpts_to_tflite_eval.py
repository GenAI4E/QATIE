"""
For every Lightning checkpoint under a directory: export ONNX → TF → TFLite, then evaluate
PSNR/SSIM (and average TFLite inference ms) with eval_tflite.evaluate_tflite() on the DPED test set.

TFLite inference runs on the CPU interpreter (LiteRT or tf.lite). Optional --skip_cuda/--skip_cpu
only control how many CSV rows are emitted per checkpoint; metrics are the same per row when eval runs once.

Run from submission/Source-Codes:
  python ckpts_to_tflite_eval.py --ckpt_dir ./ckpts --data_dir /path/to/dped --output_csv ./ckpts/tflite_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_tflite import evaluate_tflite
from src.export.to_tflite import convert_pytorch_to_tflite
from src.checkpoint_tflite_utils import (
    load_checkpoint_model_for_export,
    load_eval_test_data_if_needed,
    requested_device_names,
    resolve_output_paths,
    validate_requested_devices,
)

from src.eval.benchmark_ckpts import (
    _gather_checkpoint_paths,
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    default_ckpt_dir = _script_dir() / "ckpts"
    p = argparse.ArgumentParser(description="Convert ckpts to TFLite and evaluate with eval_tflite.evaluate_tflite.")
    p.add_argument("--ckpt_dir", type=str, default=str(default_ckpt_dir), help="Directory with .ckpt / .pt files")
    p.add_argument(
        "--full_hd",
        action="store_true",
        help="Use full-hd test data",
    )
    p.add_argument("--data_dir", type=str, required=True, help="DPED root (iphone/ …) for test eval")
    p.add_argument(
        "--tflite_root",
        type=str,
        default=None,
        help="Directory for exported TFLite trees (default: <ckpt_dir>/tflite_exports)",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="CSV path (default: <ckpt_dir>/tflite_eval_cpu_cuda.csv)",
    )
    p.add_argument(
        "--input_h",
        type=int,
        default=100,
        help="Trace height for ONNX→TF export (fixed graph at this H×W). Runtime TFLite with --dynamic accepts any H×W; eval then uses each image's native size.",
    )
    p.add_argument(
        "--input_w",
        type=int,
        default=100,
        help="Trace width for ONNX→TF export (see --input_h).",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Emit model_none.tflite with flexible H,W; output is reshaped to match input. Eval uses native per-image resolution (do not force 100×100).",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_cuda", action="store_true", help="Do not emit CSV rows with device=cuda (metrics are TFLite-on-CPU)")
    p.add_argument("--skip_cpu", action="store_true", help="Do not emit CSV rows with device=cpu")
    p.add_argument("--fullhd", action="store_true", help="Use full-size test images")
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
        default_root_name="tflite_exports",
        default_csv_name="tflite_eval_cpu_cuda.csv",
    )

    paths = _gather_checkpoint_paths(ckpt_dir)
    if not paths:
        print(f"No .ckpt or .pt files under {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    validate_requested_devices(args)

    fieldnames = [
        "checkpoint_path",
        "tflite_path",
        "ablation_model",
        "channels",
        "device",
        "export_sec",
        "eval_sec",
        "psnr",
        "ssim",
        "avg_inference_ms",
    ]
    rows: list[dict] = []

    full_hd = bool(args.full_hd or args.fullhd)
    test_data = load_eval_test_data_if_needed(
        data_dir=args.data_dir,
        skip_eval=args.skip_eval,
        full_hd=full_hd,
    )

    t_run0 = time.perf_counter()
    for ckpt_path in paths:
        print(f"\n=== {ckpt_path} ===")
        loaded = load_checkpoint_model_for_export(
            ckpt_path,
            ablation_override=args.ablation_model,
        )
        if loaded is None:
            continue

        stem = loaded.checkpoint_path.stem
        export_dir = tflite_root / stem
        try:
            t_export0 = time.perf_counter()
            tflite_path = convert_pytorch_to_tflite(
                loaded.model,
                export_dir,
                stem,
                args.input_h,
                args.input_w,
                args.dynamic,
                opset_version=args.opset,
                legacy_onnx=args.legacy_onnx,
            )
            export_sec = time.perf_counter() - t_export0
        except Exception as e:
            print(f"  TFLite export failed: {e}", file=sys.stderr)
            continue

        print(f"  TFLite: {tflite_path}  (export {export_sec:.2f}s)")

        devices = requested_device_names(args)

        if args.skip_eval:
            psnr, ssim, avg_ms = 0.0, 0.0, 0.0
            eval_sec = 0.0
        else:
            t_eval0 = time.perf_counter()
            # Static TFLite: eval at export resolution. Dynamic TFLite: do not force trace H×W — use
            # per-image native H×W so PSNR/SSIM match same-resolution I/O (see eval_tflite.evaluate_tflite).
            eval_kw = (
                {"eval_height": None, "eval_width": None}
                if args.dynamic
                else {"eval_height": args.input_h, "eval_width": args.input_w}
            )
            psnr, ssim, avg_ms = evaluate_tflite(
                tflite_path,
                test_data,
                use_xnnpack=False,
                **eval_kw,
            )
            eval_sec = time.perf_counter() - t_eval0
            print(f"  eval {eval_sec:.2f}s  PSNR={psnr:.4f}  SSIM={ssim:.4f}  avg_inf={avg_ms:.2f} ms")

        for dev_name in devices:
            rows.append(
                {
                    "checkpoint_path": str(loaded.checkpoint_path),
                    "tflite_path": str(tflite_path),
                    "ablation_model": loaded.ablation_model,
                    "channels": loaded.channels,
                    "device": dev_name,
                    "export_sec": f"{export_sec:.3f}",
                    "eval_sec": f"{eval_sec:.3f}",
                    "psnr": f"{psnr:.6f}",
                    "ssim": f"{ssim:.6f}",
                    "avg_inference_ms": f"{avg_ms:.4f}",
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
