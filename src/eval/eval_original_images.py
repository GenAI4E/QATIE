"""
Per-phone TFLite conversion and evaluation on original_images/test.

For every Lightning checkpoint under --ckpt_dir: export a static TFLite model
per phone (iphone / blackberry / sony) at that phone's native resolution, then
evaluate PSNR / SSIM against the canon ground-truth images.

Phone resolutions (WxH):
    iphone:     2048 x 1536
    blackberry: 3120 x 3120
    sony:       2592 x 1944
    canon (GT): 3648 x 2432

Each TFLite has a fixed input shape matching the phone it targets. Canon images
are resized to match the model output (= phone input size) for comparison.

Run from submission/Source-Codes (using the export_env conda environment):
    python eval_original_images.py \
        --ckpt_dir ./ckpts \
        --test_dir /path/to/mixedmodel/dataset/original_images/test \
        --output_dir ./ckpts/phone_eval

Output structure:
    <output_dir>/
        iphone/
            <ckpt_stem>/<stem>.tflite     # input [1,1536,2048,3]
            eval_results.csv
        blackberry/
            <ckpt_stem>/<stem>.tflite     # input [1,3120,3120,3]
            eval_results.csv
        sony/
            <ckpt_stem>/<stem>.tflite     # input [1,1944,2592,3]
            eval_results.csv
        combined_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_tflite_utils import load_checkpoint_model_for_export
from src.eval.eval_tflite import evaluate_tflite
from src.export.to_tflite import convert_pytorch_to_tflite

from src.eval.benchmark_ckpts import (
    _gather_checkpoint_paths,
)

# PHONE_CONFIGS = {
#     "iphone":     {"width": 2048, "height": 1536},
#     "blackberry": {"width": 3120, "height": 3120},
#     "sony":       {"width": 2592, "height": 1944},
# }
PHONE_CONFIGS = {
    "iphone":     {"width": 2048, "height": 1536}, # same
    "blackberry": {"width": 2560, "height": 1440}, # 2K HD
    "sony":       {"width": 2560, "height": 1440}, # 2K HD
}

def load_phone_canon_pairs(
    test_dir: str | Path,
    phone_name: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load (phone_image, canon_image) pairs matched by filename.

    Returns float32 HWC arrays in [0, 1].
    """
    test_dir = Path(test_dir)
    phone_dir = test_dir / phone_name
    canon_dir = test_dir / "canon"
    if not phone_dir.is_dir():
        raise FileNotFoundError(f"Phone directory not found: {phone_dir}")
    if not canon_dir.is_dir():
        raise FileNotFoundError(f"Canon directory not found: {canon_dir}")

    phone_files = {
        Path(p).name: p
        for p in sorted(glob(str(phone_dir / "*.jpg")))
    }
    canon_files = {
        Path(p).name: p
        for p in sorted(glob(str(canon_dir / "*.jpg")))
    }
    common = sorted(set(phone_files) & set(canon_files))
    if not common:
        raise ValueError(
            f"No overlapping filenames between {phone_dir} and {canon_dir}"
        )

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for fname in tqdm(common, desc=f"Loading {phone_name}+canon pairs"):
        phone_img = np.array(Image.open(phone_files[fname])).astype(np.float32) / 255.0
        canon_img = np.array(Image.open(canon_files[fname])).astype(np.float32) / 255.0
        pairs.append((phone_img, canon_img))
    return pairs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_ckpt_dir = script_dir / "ckpts"
    default_output_dir = script_dir / "ckpts" / "phone_eval"

    p = argparse.ArgumentParser(
        description="Per-phone TFLite conversion + eval on original_images/test."
    )
    p.add_argument(
        "--ckpt_dir", type=str, default=str(default_ckpt_dir),
        help="Directory with .ckpt / .pt files",
    )
    p.add_argument(
        "--test_dir", type=str, required=True,
        help="Path to original_images/test (with iphone/, blackberry/, sony/, canon/ subdirs)",
    )
    p.add_argument(
        "--output_dir", type=str, default=str(default_output_dir),
        help="Root output directory for per-phone TFLite exports and CSVs",
    )
    p.add_argument(
        "--phones", type=str, nargs="+",
        default=list(PHONE_CONFIGS.keys()),
        choices=list(PHONE_CONFIGS.keys()),
        help="Which phones to evaluate (default: all)",
    )
    p.add_argument("--skip_eval", action="store_true", help="Export only, skip evaluation")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    p.add_argument(
        "--legacy-onnx", dest="legacy_onnx", action="store_true", default=False,
        help="TorchScript ONNX export (dynamo=False)",
    )
    p.add_argument("--ablation_model", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not ckpt_dir.is_dir():
        print(f"Error: not a directory: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    if not test_dir.is_dir():
        print(f"Error: test_dir not found: {test_dir}", file=sys.stderr)
        sys.exit(1)

    paths = _gather_checkpoint_paths(ckpt_dir)
    if not paths:
        print(f"No .ckpt or .pt files under {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} checkpoint(s) under {ckpt_dir}")

    phone_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    if not args.skip_eval:
        for phone in args.phones:
            phone_data[phone] = load_phone_canon_pairs(test_dir, phone)
            print(f"  {phone}: {len(phone_data[phone])} pair(s)")

    fieldnames = [
        "phone",
        "checkpoint_path",
        "tflite_path",
        "ablation_model",
        "channels",
        "input_h",
        "input_w",
        "export_sec",
        "eval_sec",
        "psnr",
        "ssim",
        "avg_inference_ms",
    ]
    all_rows: list[dict] = []
    per_phone_rows: dict[str, list[dict]] = {ph: [] for ph in args.phones}

    t_total = time.perf_counter()

    for ckpt_path in paths:
        print(f"\n{'='*60}\n  Checkpoint: {ckpt_path}\n{'='*60}")
        loaded = load_checkpoint_model_for_export(
            ckpt_path,
            ablation_override=args.ablation_model,
        )
        if loaded is None:
            continue
        stem = loaded.checkpoint_path.stem
        print(f"  ablation={loaded.ablation_model}  channels={loaded.channels}")

        for phone in args.phones:
            cfg = PHONE_CONFIGS[phone]
            ph_h, ph_w = cfg["height"], cfg["width"]
            print(f"\n  --- {phone} ({ph_w}x{ph_h}) ---")

            phone_export_dir = output_dir / phone / stem
            t0 = time.perf_counter()
            try:
                tflite_path = convert_pytorch_to_tflite(
                    loaded.model, phone_export_dir, stem,
                    ph_h, ph_w,
                    dynamic_shape=False,
                    opset_version=args.opset,
                    legacy_onnx=args.legacy_onnx,
                )
            except Exception as e:
                print(f"  Export failed for {phone}: {e}", file=sys.stderr)
                continue
            export_sec = time.perf_counter() - t0
            print(f"  TFLite: {tflite_path}  ({export_sec:.1f}s)")

            psnr = ssim = avg_ms = 0.0
            eval_sec = 0.0
            if not args.skip_eval:
                t0 = time.perf_counter()
                psnr, ssim, avg_ms = evaluate_tflite(
                    tflite_path,
                    phone_data[phone],
                    use_xnnpack=False,
                    eval_height=ph_h,
                    eval_width=ph_w,
                )
                eval_sec = time.perf_counter() - t0
                print(f"  PSNR={psnr:.4f}  SSIM={ssim:.4f}  avg_inf={avg_ms:.2f}ms  ({eval_sec:.1f}s)")

            row = {
                "phone": phone,
                "checkpoint_path": str(ckpt_path),
                "tflite_path": str(tflite_path),
                "ablation_model": loaded.ablation_model,
                "channels": loaded.channels,
                "input_h": ph_h,
                "input_w": ph_w,
                "export_sec": f"{export_sec:.3f}",
                "eval_sec": f"{eval_sec:.3f}",
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim:.6f}",
                "avg_inference_ms": f"{avg_ms:.4f}",
            }
            all_rows.append(row)
            per_phone_rows[phone].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for phone in args.phones:
        phone_csv = output_dir / phone / "eval_results.csv"
        phone_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(phone_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_phone_rows[phone])
        print(f"\nWrote {len(per_phone_rows[phone])} row(s) to {phone_csv}")

    combined_csv = output_dir / "combined_results.csv"
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} row(s) to {combined_csv}")

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
