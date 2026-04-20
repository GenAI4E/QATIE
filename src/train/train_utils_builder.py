"""
train_utils_builder.py — shared registries and factory functions for training.

Imported by train.py, train_qat.py, and any future training scripts.
"""

import sys
from pathlib import Path

import torch.optim as optim

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.loss import CombinedLoss, CombinedLossV2, CombinedLossV3


# ======================== Registries ========================

# Maps loss_version -> (class, kwarg_keys, component_names, description)
LOSS_REGISTRY = {
    1: {
        'cls': CombinedLoss,
        'weight_keys': ['weight_psnr', 'weight_msssim', 'weight_outlier'],
        'component_names': ['psnr_loss', 'msssim_loss', 'outlier_loss'],
        'description': 'PSNR + MSSSIM + OutlierAwareLoss',
    },
    2: {
        'cls': CombinedLossV2,
        'weight_keys': ['weight_psnr', 'weight_cosine', 'weight_outlier'],
        'component_names': ['psnr_loss', 'cosine_loss', 'outlier_loss'],
        'description': 'PSNR + CosineSim + OutlierAwareLoss',
    },
    3: {
        'cls': CombinedLossV3,
        'weight_keys': ['weight_msmse', 'weight_msssim', 'weight_perceptual'],
        'component_names': ['msmse_loss', 'msssim_loss', 'perceptual_loss'],
        'description': 'MSMSE + MSSSIM + PerceptualLoss',
    }
}

SCHEDULER_REGISTRY = {
    'cosine_warm_restarts': 'CosineAnnealingWarmRestarts',
    'cosine_warmup': 'LinearWarmup + CosineAnnealingLR',
}


# ======================== Factories ========================

def build_criterion(loss_version: int, config):
    """Factory function to build loss from version number."""
    if loss_version not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_version={loss_version}. Choose from {list(LOSS_REGISTRY)}.")
    info = LOSS_REGISTRY[loss_version]
    kwargs = {k: getattr(config, k) for k in info['weight_keys']}
    return info['cls'](**kwargs, resize=True, return_loss_components=True)


def build_scheduler(optimizer, config):
    """Factory function to build LR scheduler from config."""
    scheduler_type = config.scheduler_type
    if scheduler_type == 'cosine_warm_restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
        )
    elif scheduler_type == 'cosine_warmup':
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.warmup_start_factor, total_iters=config.warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.num_epochs - config.warmup_epochs), eta_min=config.eta_min
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
        )
    else:
        raise ValueError(f"Unknown scheduler_type={scheduler_type!r}. Choose from {list(SCHEDULER_REGISTRY)}.")
