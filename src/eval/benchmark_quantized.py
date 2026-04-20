"""
Benchmark quantized artifacts under a directory (same spirit as benchmark_ckpts.py):

- INT8 PyTorch models saved as *_int8.pth (output of quantize.py / convert_fx)
- Optional .tflite files (same DPED test split as quantize.evaluate_model for PSNR/SSIM)

Metrics: file size, parameter memory (PyTorch), mean/variance of PSNR/SSIM over
--bench_trials full test passes, forward latency (PyTorch: synthetic fixed size;
TFLite: optional synthetic timing on a representative patch).

If ``torch.load`` on ``*_int8.pth`` fails (common: FX GraphModule pickle incompatibility),
the script rebuilds INT8 from a Lightning ``.ckpt``: use ``--rebuild_ckpt`` or
``--rebuild_ckpt_dir``, or rely on auto-pick from ``.../ckpts/`` when the int8 file lives under
``.../int8/with_qat/`` or ``.../int8/no_qat/`` (prefers ``*with_qat*cN_*.ckpt`` / ``*no_qat*cN_*.ckpt``).

Run from this directory:
  python benchmark_quantized.py --model_dir ./results/int8 --data_dir /path/to/dped \\
    --output_csv ./results/int8/benchmark_quantized.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.checkpoint_loading import ABLATION_CHOICES
from src.eval.benchmark_ckpts import (
    _mean_sample_var,
    benchmark_forward_ms,
    checkpoint_file_mb,
    param_mib,
)

# hybrid_base_c32_int8.pth → ablation + width
_INT8_STEM_RE = re.compile(r"^(?P<ablation>hybrid_[a-z_]+)_c(?P<channels>\d+)_int8$")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_int8_stem(stem: str) -> tuple[str, int] | None:
    m = _INT8_STEM_RE.match(stem)
    if not m:
        return None
    return m.group("ablation"), int(m.group("channels"))


def _pick_newest_ckpt_for_channels(
    ckpt_dir: Path,
    channels: int,
    *,
    int8_parent: str | None = None,
) -> Path | None:
    """
    Prefer checkpoints that match the int8 export folder (with_qat vs no_qat) so we do not
    pick an unrelated *c32_*.ckpt (e.g. channel_ablation) when rebuilding.
    """
    patterns: list[str]
    if int8_parent == "with_qat":
        patterns = [f"*with_qat*c{channels}_*.ckpt", f"*c{channels}_*.ckpt"]
    elif int8_parent == "no_qat":
        patterns = [f"*no_qat*c{channels}_*.ckpt", f"*c{channels}_*.ckpt"]
    else:
        patterns = [f"*c{channels}_*.ckpt"]

    for pat in patterns:
        matches = sorted(
            ckpt_dir.glob(pat),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


def _candidate_ckpt_dirs_near_int8(model_path: Path) -> list[Path]:
    """
    If model is under .../<run>/int8/<subdir>/file.pth, search for .ckpt under:
      <run>/checkpoints/  then  <run>/  (e.g. ckpts/*.ckpt next to int8/).
    """
    p = model_path.resolve()
    if p.parent.parent.name != "int8":
        return []
    base = p.parent.parent.parent
    out: list[Path] = []
    for sub in ("checkpoints", ""):
        cand = base / sub if sub else base
        if cand.is_dir():
            out.append(cand)
    return out


def load_int8_torch_model(
    model_path: Path,
    *,
    rebuild_ckpt: str | None,
    rebuild_ckpt_dir: str | None,
    ablation_override: str | None,
    legacy_qat_graph: bool,
    data_dir: str | None,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load pickled INT8 GraphModule, or rebuild from a Lightning ckpt if unpickling fails
    (FX GraphModule pickles often break across PyTorch versions).
    """
    from src.models.model_builder import infer_channels_from_checkpoint
    from src.export.quantize import convert_checkpoint_to_int8_model

    try:
        m = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(m, torch.nn.Module):
            return m
        print(f"  Warning: {model_path.name} is not an nn.Module after load; rebuilding...", file=sys.stderr)
    except Exception as e:
        print(f"  torch.load failed ({type(e).__name__}: {e})", file=sys.stderr)
        print("  Rebuilding INT8 from Lightning checkpoint instead...", file=sys.stderr)

    parsed = parse_int8_stem(model_path.stem)
    int8_parent = model_path.parent.name if model_path.parent.name in ("with_qat", "no_qat") else None
    ckpt_path: Path | None = None
    if rebuild_ckpt:
        ckpt_path = Path(rebuild_ckpt).expanduser().resolve()
    elif rebuild_ckpt_dir:
        ckpt_dir = Path(rebuild_ckpt_dir).expanduser().resolve()
        if not parsed:
            raise RuntimeError(
                f"Cannot parse ablation/channels from {model_path.name!r}; "
                "use --rebuild_ckpt with a single .ckpt, or name files like hybrid_base_c32_int8.pth."
            )
        _, ch = parsed
        ckpt_path = _pick_newest_ckpt_for_channels(ckpt_dir, ch, int8_parent=int8_parent)
    else:
        if parsed:
            _, ch = parsed
            for inferred in _candidate_ckpt_dirs_near_int8(model_path):
                ckpt_path = _pick_newest_ckpt_for_channels(inferred, ch, int8_parent=int8_parent)
                if ckpt_path is not None:
                    print(f"  Using inferred checkpoint dir: {inferred}", file=sys.stderr)
                    break

    if ckpt_path is None or not ckpt_path.is_file():
        raise RuntimeError(
            "Cannot rebuild INT8: no usable Lightning checkpoint.\n"
            "  Pass --rebuild_ckpt /path/to/run.ckpt or --rebuild_ckpt_dir /path/to/checkpoints "
            "(newest *c<channels>_*.ckpt matching the int8 filename).\n"
            "  Filenames must look like hybrid_base_c32_int8.pth so channels match."
        )

    if parsed:
        ablation, ch = parsed
    else:
        ch = int(infer_channels_from_checkpoint(str(ckpt_path)))
        ablation = "hybrid_base"

    if ablation_override:
        name = ablation_override.lower()
        if name not in ABLATION_CHOICES:
            raise ValueError(f"--ablation_model {name!r} must be one of {ABLATION_CHOICES}")
        ablation = name

    print(f"  Rebuilding from {ckpt_path} (ablation_model={ablation}, channels={ch})...")
    return convert_checkpoint_to_int8_model(
        str(ckpt_path),
        ablation_model=ablation,
        channels=ch,
        legacy_qat_graph=legacy_qat_graph,
        device=device,
        data_dir=data_dir,
        save_path=None,
    )


def _gather_quantized_paths(root: Path, tflite: bool) -> list[Path]:
    """Collect *_int8.pth under root; optionally also *.tflite."""
    paths: list[Path] = []
    paths.extend(sorted(root.rglob("*_int8.pth")))
    if tflite:
        paths.extend(sorted(root.rglob("*.tflite")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def evaluate_tflite_same_split_as_quantize(
    tflite_path: str | Path,
    data_dir: str,
    full_hd: bool = False,
    num_workers: int = 4,
    verbose: bool = False,
) -> tuple[float, float]:
    from tqdm import tqdm
    from src.data.dped_dataset import DPEDDataModule
    from src.eval.eval_tflite import compute_psnr, compute_ssim, load_tflite_model, run_tflite_inference
    
    data_path_list = [(phone, Path(data_dir) / phone) for phone in os.listdir(data_dir)]
    datamodule = DPEDDataModule(data_dir=data_path_list, patch_data=not full_hd, batch_size=1, num_workers=num_workers)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    interpreter, input_details, output_details = load_tflite_model(str(tflite_path))
    input_shape = list(input_details[0]["shape"])
    is_nchw = len(input_shape) == 4 and input_shape[1] == 3

    psnr_list: list[float] = []
    ssim_list: list[float] = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Eval TFLite", disable=not verbose):
            inp = inputs.numpy()
            tgt = targets.numpy()
            if inp.ndim == 4:
                inp = inp[0]
            if tgt.ndim == 4:
                tgt = tgt[0]
            # NCHW [3,H,W] from DataLoader
            if is_nchw:
                input_data = inp.astype(np.float32)[np.newaxis, ...]
            else:
                input_data = np.transpose(inp, (1, 2, 0))[np.newaxis, ...].astype(np.float32)

            if list(input_data.shape) != list(input_details[0]["shape"]):
                interpreter.resize_tensor_input(input_details[0]["index"], list(input_data.shape))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

            output = run_tflite_inference(interpreter, input_details, output_details, input_data)
            output = output.squeeze(0)
            if output.ndim == 3 and output.shape[0] == 3:
                output = np.transpose(output, (1, 2, 0))
            output = np.clip(output, 0, 1)

            gt = np.transpose(tgt, (1, 2, 0)) if tgt.ndim == 3 and tgt.shape[0] == 3 else tgt
            psnr_list.append(compute_psnr(output, gt.astype(np.float64)))
            ssim_list.append(compute_ssim(output, gt.astype(np.float64)))

    if verbose:
        print(f"Results on {len(psnr_list)} images:")
        print(f"  PSNR: {np.mean(psnr_list):.4f} dB")
        print(f"  SSIM: {np.mean(ssim_list):.4f}")
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def benchmark_tflite_forward_ms(
    tflite_path: str | Path,
    height: int,
    width: int,
    warmup: int,
    runs: int,
    bench_trials: int,
) -> list[float]:
    """Synthetic timing: one forward per trial mean (like benchmark_forward_ms for PyTorch)."""
    from src.eval.eval_tflite import load_tflite_model, run_tflite_inference

    ms_vals: list[float] = []
    for _ in range(bench_trials):
        interpreter, input_details, output_details = load_tflite_model(str(tflite_path))
        sh = list(input_details[0]["shape"])
        is_nchw = len(sh) == 4 and sh[1] == 3
        if is_nchw:
            x = np.random.randn(1, 3, height, width).astype(np.float32)
        else:
            x = np.random.randn(1, height, width, 3).astype(np.float32)

        if list(x.shape) != sh:
            interpreter.resize_tensor_input(input_details[0]["index"], list(x.shape))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

        for _ in range(warmup):
            run_tflite_inference(interpreter, input_details, output_details, x)

        times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run_tflite_inference(interpreter, input_details, output_details, x)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms_vals.append(float(np.mean(times)))
    return ms_vals


def parse_args() -> argparse.Namespace:
    default_dir = _script_dir() / "ckpts"
    p = argparse.ArgumentParser(
        description="Benchmark INT8 PyTorch and/or TFLite models: size, PSNR, SSIM, latency."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default=str(default_dir),
        help="Root directory to scan recursively for *_int8.pth (and .tflite if --include_tflite)",
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
        default="cpu",
        choices=("cpu", "cuda"),
        help="PyTorch INT8 eval/timing device (qnnpack INT8 is typically CPU; TFLite ignores this)",
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
        help="Repeat full eval+timing N times; report mean and sample variance",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Write results CSV (default: <model_dir>/benchmark_quantized.csv)",
    )
    p.add_argument("--skip_eval", action="store_true", help="Skip PSNR/SSIM (size + timing only)")
    p.add_argument("--skip_timing", action="store_true", help="Skip latency benchmark")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for evaluation")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument(
        "--include_tflite",
        action="store_true",
        help="Also evaluate *.tflite under model_dir (requires TensorFlow)",
    )
    p.add_argument(
        "--rebuild_ckpt",
        type=str,
        default=None,
        help="Lightning .ckpt used to re-run convert_fx when *_int8.pth fails to torch.load",
    )
    p.add_argument(
        "--rebuild_ckpt_dir",
        type=str,
        default=None,
        help="Directory to pick newest *c<channels>_*.ckpt (channels from int8 filename)",
    )
    p.add_argument(
        "--ablation_model",
        type=str,
        default=None,
        metavar="NAME",
        help=f"When rebuilding and filename does not parse, architecture name (one of {ABLATION_CHOICES})",
    )
    p.add_argument(
        "--legacy_qat_graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When rebuilding, match quantize.py legacy FX graph (default: on; use --no-legacy-qat-graph if not)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"Error: model_dir is not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_eval and not args.data_dir:
        print("Error: --data_dir is required unless --skip_eval", file=sys.stderr)
        sys.exit(1)

    paths = _gather_quantized_paths(model_dir, tflite=args.include_tflite)
    if not paths:
        print(
            f"No *_int8.pth files found under {model_dir}"
            + (" (and no .tflite with --include_tflite)" if args.include_tflite else ""),
            file=sys.stderr,
        )
        sys.exit(1)

    out_csv = args.output_csv
    if out_csv is None:
        out_csv = str(model_dir / "benchmark_quantized.csv")
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; use --device cpu", file=sys.stderr)
        sys.exit(1)

    from src.eval.eval_pytorch import evaluate_model

    fieldnames = [
        "model_path",
        "kind",
        "file_mb",
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
    n_trials = max(1, int(args.bench_trials))

    for model_path in paths:
        print(f"\n=== {model_path} ===", flush=True)
        kind = "tflite" if model_path.suffix.lower() == ".tflite" else "int8_pth"
        ck_mb = checkpoint_file_mb(model_path)

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        ms_vals: list[float] = []

        p_mib_s = ""
        if kind == "int8_pth":
            try:
                model = load_int8_torch_model(
                    model_path,
                    rebuild_ckpt=args.rebuild_ckpt,
                    rebuild_ckpt_dir=args.rebuild_ckpt_dir,
                    ablation_override=args.ablation_model,
                    legacy_qat_graph=bool(args.legacy_qat_graph),
                    data_dir=args.data_dir,
                    device=device,
                )
            except Exception as e:
                print(f"  Skipping: {e}", file=sys.stderr)
                continue
            p_mib = param_mib(model)
            p_mib_s = f"{p_mib:.6f}"
            print(f"  kind={kind}  file_mb={ck_mb:.4f}  param_mib={p_mib_s}")

            if not args.skip_eval:
                assert args.data_dir is not None
                for t in range(n_trials):
                    psnr_f, ssim_f = evaluate_model(
                        model,
                        args.data_dir,
                        batch_size=1,
                        num_workers=args.num_workers,
                        device=device,
                        verbose=args.verbose,
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
        else:
            print(f"  kind={kind}  file_mb={ck_mb:.4f}  (param_mib N/A for TFLite)")
            if not args.skip_eval:
                assert args.data_dir is not None
                for t in range(n_trials):
                    psnr_f, ssim_f = evaluate_tflite_same_split_as_quantize(
                        model_path,
                        args.data_dir,
                        full_hd=args.full_hd,
                        num_workers=args.num_workers,
                        verbose=args.verbose,
                    )
                    psnr_vals.append(psnr_f)
                    ssim_vals.append(ssim_f)
                    print(f"  trial {t + 1}/{n_trials}  PSNR={psnr_f:.6f}  SSIM={ssim_f:.6f}")

            if not args.skip_timing:
                ms_vals = benchmark_tflite_forward_ms(
                    model_path,
                    args.timing_h,
                    args.timing_w,
                    args.warmup,
                    args.runs,
                    n_trials,
                )
                for t, ms in enumerate(ms_vals):
                    print(
                        f"  timing trial {t + 1}/{n_trials}  forward_ms={ms:.6f}  "
                        f"(synthetic, batch=1x3x{args.timing_h}x{args.timing_w} or NHWC, "
                        f"warmup={args.warmup}, runs={args.runs})"
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
            print(f"  forward_ms mean={fwd_m} var={fwd_var_s}  ({n_trials} trials)")

        rows.append(
            {
                "model_path": str(model_path),
                "kind": kind,
                "file_mb": f"{ck_mb:.6f}",
                "param_mib": p_mib_s,
                "bench_trials": n_trials,
                "psnr_mean": psnr_m,
                "psnr_var": psnr_var_s,
                "ssim_mean": ssim_m,
                "ssim_var": ssim_var_s,
                "forward_ms_mean": fwd_m,
                "forward_ms_var": fwd_var_s,
                "timing_h": args.timing_h,
                "timing_w": args.timing_w,
                "device": str(device) if kind == "int8_pth" else "tflite_cpu",
            }
        )

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
