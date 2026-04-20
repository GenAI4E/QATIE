"""
Evaluate a PyTorch model (FP32 or INT8) on the DPED test split.

Reports mean PSNR (dB) and SSIM over all test image pairs.

Run from this directory:
  python eval_pytorch.py --ckpt_path ./ckpts/model.ckpt --data_dir /path/to/dped
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dped_dataset import DPEDDataModule


def evaluate_model(
    model,
    data_dir,
    batch_size=1,
    num_workers=4,
    device=torch.device("cpu"),
    verbose=True,
    full_hd: bool = False,
):
    """Evaluate model on the test split of a DPED-style dataset, reporting PSNR and SSIM."""
    from tqdm import tqdm
    from skimage.metrics import structural_similarity

    data_path_list = [(phone, Path(data_dir) / phone) for phone in os.listdir(data_dir)]
    datamodule = DPEDDataModule(data_dir=data_path_list, patch_data=not full_hd, batch_size=batch_size, num_workers=num_workers)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    model.eval().to(device)
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", disable=not verbose):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            targets = targets.numpy()
            for pred, gt in zip(outputs, targets):
                mse = np.mean((pred - gt) ** 2)
                psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
                psnr_list.append(psnr)

                pred_u8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
                gt_u8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
                ssim = np.mean(
                    [
                        structural_similarity(
                            gt_u8[:, :, c],
                            pred_u8[:, :, c],
                            gaussian_weights=True,
                            sigma=1.5,
                            data_range=255,
                        )
                        for c in range(3)
                    ]
                )
                ssim_list.append(ssim)

    if verbose:
        print(f"Results on {len(psnr_list)} images:")
        print(f"  PSNR: {np.mean(psnr_list):.4f} dB")
        print(f"  SSIM: {np.mean(ssim_list):.4f}")
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


if __name__ == "__main__":
    from src.models.model_builder import build_model, infer_channels_from_checkpoint, load_checkpoint_weights

    parser = argparse.ArgumentParser(description="Evaluate a PyTorch checkpoint on DPED test set.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to Lightning .ckpt or .pt checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="DPED dataset root (contains iphone/ etc.)")
    parser.add_argument("--channels", type=int, default=None, help="Base channel width (inferred from ckpt if omitted)")
    parser.add_argument("--ablation_model", type=str, default="hybrid_base", help="Model variant name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full_hd", action="store_true", help="Use full-hd test data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    channels = args.channels
    if channels is None:
        try:
            channels = infer_channels_from_checkpoint(args.ckpt_path)
        except Exception as e:
            print(f"Could not infer channels ({e}); using default 32.")
            channels = 32

    model = build_model(channels, model_name=args.ablation_model)
    model = load_checkpoint_weights(model, args.ckpt_path)
    model.eval()

    device = torch.device(args.device)
    psnr, ssim = evaluate_model(
        model,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        full_hd=args.full_hd,
    )
    print(f"\nFinal — PSNR: {psnr:.4f} dB  SSIM: {ssim:.4f}")
