"""
Benchmark Lightning checkpoints under a directory: file size, parameter memory,
mean/variance of PSNR/SSIM on DPED test (same as quantize.evaluate_model) and forward latency (ms)
over repeated trials (default 5).

Run from this directory:
  python benchmark_ckpts.py --ckpt_dir ./ckpts --data_dir /path/to/dped/dped --output_csv ./ckpts/benchmark_results.csv
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
from src.models.model_builder import (
    infer_channels_from_checkpoint,
    is_fx_quantized_full_model_checkpoint,
    load_torch_checkpoint_trusted,
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _gather_checkpoint_paths(ckpt_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.ckpt", "*.pt", "*.pth"):
        paths.extend(sorted(ckpt_dir.rglob(ext)))
    # De-dupe same file
    seen = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def param_mib(model: torch.nn.Module) -> float:
    """FP32 parameter bytes → MiB (same convention as train_qat param count)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024**2)


def checkpoint_file_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024**2)


def benchmark_forward_ms(
    model: torch.nn.Module,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    runs: int,
) -> float:
    model.eval()
    x = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def _mean_sample_var(vals: list[float]) -> tuple[float, float]:
    """Unbiased sample variance (ddof=1); variance is 0.0 if len(vals) < 2."""
    arr = np.asarray(vals, dtype=np.float64)
    m = float(np.mean(arr))
    if arr.size < 2:
        return m, 0.0
    return m, float(np.var(arr, ddof=1))


def parse_args() -> argparse.Namespace:
    default_ckpt_dir = _script_dir() / "ckpts"
    p = argparse.ArgumentParser(description="Benchmark checkpoints: size, PSNR, SSIM, runtime.")
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(default_ckpt_dir),
        help="Directory to scan recursively for .ckpt and .pt files",
    )
    p.add_argument(
        "--full_hd",
        action="store_true",
        help="Use full-hd test data",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="DPED dataset root (iphone/ …). Required unless --skip_eval.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    p.add_argument("--timing_h", type=int, default=1088, help="Latency benchmark input height")
    p.add_argument("--timing_w", type=int, default=1920, help="Latency benchmark input width")
    p.add_argument("--warmup", type=int, default=20, help="Warmup forward passes before timing")
    p.add_argument("--runs", type=int, default=100, help="Timed forward passes per timing trial")
    p.add_argument(
        "--bench_trials",
        type=int,
        default=5,
        metavar="N",
        help="Repeat full eval+timing N times; report mean and sample variance of PSNR, SSIM, runtime",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Write results CSV (default: <ckpt_dir>/benchmark_results.csv)",
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip PSNR/SSIM on test set (size + timing only)",
    )
    p.add_argument("--skip_timing", action="store_true", help="Skip forward latency benchmark")
    p.add_argument(
        "--ablation_model",
        type=str,
        default=None,
        help="Force model variant (skip auto-pick). One of hybrid_* names.",
    )
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for evaluation")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.is_dir():
        print(f"Error: ckpt_dir is not a directory: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_eval and not args.data_dir:
        print("Error: --data_dir is required unless --skip_eval", file=sys.stderr)
        sys.exit(1)

    paths = _gather_checkpoint_paths(ckpt_dir)
    if not paths:
        print(f"No .ckpt or .pt files found under {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    out_csv = args.output_csv
    if out_csv is None:
        out_csv = str(ckpt_dir / "benchmark_results.csv")
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; use --device cpu", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "checkpoint_path",
        "ablation_model",
        "channels",
        "checkpoint_file_mb",
        "param_mib",
        "bench_trials",
        "psnr_mean",
        "psnr_var",
        "ssim_mean",
        "ssim_var",
        "forward_ms_mean",
        "forward_ms_var",
        "timing_h",
        "timing_w",
        "device",
    ]

    rows: list[dict] = []
    for ckpt_path in paths:
        print(f"\n=== {ckpt_path} ===")
        if is_fx_quantized_full_model_checkpoint(ckpt_path):
            print(
                "  Skipping: FX INT8 GraphModule .pth (use Lightning .ckpt for benchmarking/export).",
                file=sys.stderr,
            )
            continue
        raw = load_torch_checkpoint_trusted(ckpt_path)
        hp_ch, hp_ab = _try_hparams_channels_ablation(raw if isinstance(raw, dict) else {})

        try:
            channels = int(hp_ch) if hp_ch is not None else infer_channels_from_checkpoint(str(ckpt_path))
        except Exception as e:
            print(f"  Skipping: could not infer channels: {e}", file=sys.stderr)
            continue

        try:
            stripped = extract_stripped_model_state_dict(ckpt_path)
        except ValueError as e:
            print(f"  Skipping: {e}", file=sys.stderr)
            continue
        if not stripped:
            print(f"  Skipping: no model.* keys in checkpoint", file=sys.stderr)
            continue

        override = args.ablation_model
        if override is None and hp_ab and hp_ab in ABLATION_CHOICES:
            override = hp_ab
        if override is None:
            fn_ab = _ablation_from_filename(ckpt_path)
            if fn_ab and fn_ab in ABLATION_CHOICES:
                override = fn_ab

        try:
            ab_name, model = pick_ablation_and_load(
                ckpt_path, channels, stripped, override
            )
        except Exception as e:
            print(f"  Skipping load: {e}", file=sys.stderr)
            continue

        ck_mb = checkpoint_file_mb(ckpt_path)
        p_mib = param_mib(model)
        print(f"  channels={channels}  ablation_model={ab_name}")
        print(f"  checkpoint_file_mb={ck_mb:.4f}  param_mib={p_mib:.4f}")

        n_trials = max(1, int(args.bench_trials))
        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        ms_vals: list[float] = []

        if not args.skip_eval:
            assert args.data_dir is not None
            from src.eval.eval_pytorch import evaluate_model

            for t in range(n_trials):
                psnr_f, ssim_f = evaluate_model(
                    model,
                    args.data_dir,
                    batch_size=1,
                    num_workers=args.num_workers,
                    device=device,
                    verbose=args.verbose,
                    full_hd=args.full_hd,
                )
                psnr_vals.append(psnr_f)
                ssim_vals.append(ssim_f)
                print(f"  trial {t + 1}/{n_trials}  PSNR={psnr_f:.6f}  SSIM={ssim_f:.6f}")

        if not args.skip_timing:
            model.to(device)
            for t in range(n_trials):
                ms = benchmark_forward_ms(
                    model, device, args.timing_h, args.timing_w, args.warmup, args.runs
                )
                ms_vals.append(ms)
                print(
                    f"  timing trial {t + 1}/{n_trials}  forward_ms={ms:.6f}  "
                    f"(batch=1x3x{args.timing_h}x{args.timing_w}, warmup={args.warmup}, runs={args.runs})"
                )

        psnr_m = psnr_var_s = ""
        ssim_m = ssim_var_s = ""
        if psnr_vals:
            pm, pv = _mean_sample_var(psnr_vals)
            sm, sv = _mean_sample_var(ssim_vals)
            psnr_m, psnr_var_s = f"{pm:.6f}", f"{pv:.8f}"
            ssim_m, ssim_var_s = f"{sm:.6f}", f"{sv:.8f}"
            print(
                f"  PSNR mean={psnr_m} var={psnr_var_s}  |  "
                f"SSIM mean={ssim_m} var={ssim_var_s}  ({n_trials} trials)"
            )

        fwd_m = fwd_var_s = ""
        if ms_vals:
            mm, mv = _mean_sample_var(ms_vals)
            fwd_m, fwd_var_s = f"{mm:.6f}", f"{mv:.8f}"
            print(f"  forward_ms mean={fwd_m} var={fwd_var_s}  ({n_trials} trials, device={device})")

        rows.append(
            {
                "checkpoint_path": str(ckpt_path),
                "ablation_model": ab_name,
                "channels": channels,
                "checkpoint_file_mb": f"{ck_mb:.6f}",
                "param_mib": f"{p_mib:.6f}",
                "bench_trials": n_trials,
                "psnr_mean": psnr_m,
                "psnr_var": psnr_var_s,
                "ssim_mean": ssim_m,
                "ssim_var": ssim_var_s,
                "forward_ms_mean": fwd_m,
                "forward_ms_var": fwd_var_s,
                "timing_h": args.timing_h,
                "timing_w": args.timing_w,
                "device": str(device),
            }
        )

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
