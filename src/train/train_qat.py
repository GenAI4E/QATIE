import os
import sys
from pathlib import Path
import argparse

# Ensure `from src...` imports work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
import torch.ao.quantization.quantize_fx as quantize_fx
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import wandb

from src.data.dped_dataset import DPEDDataModule
from src.models.model_builder import build_model, build_qat_qconfig_mapping
from src.data.data_aug import GPUAugmentation
from src.train.train_utils_builder import (
    LOSS_REGISTRY,
    SCHEDULER_REGISTRY,
    build_criterion,
    build_scheduler,
)

import dotenv

dotenv.load_dotenv()
torch.autograd.set_detect_anomaly(True)

class Config: # TODO: update to yaml config or anything
    # Model
    channels = 12
    ablation_model = "hybrid_base"
    # Loss
    loss_version = 1         # 1=PSNR+MSSSIM+Outlier, 2=PSNR+Cosine+Outlier
    weight_msssim = 500.0
    weight_outlier = 1.0
    weight_psnr = 2.0
    weight_cosine = 1.0
    weight_perceptual = 1.0
    # Optimizer
    learning_rate = 1e-4
    # Scheduler
    T_0 = 20
    T_mult = 2
    eta_min = 1e-6
    # Train
    num_epochs = 50
    precision = "16-mixed"
    accumulate_grad_batches = 2
    limit_train_batches = 0.1
    limit_val_batches = 1.0
    gradient_clip_val = 1.0
    gradient_clip_algorithm = "norm"
    # Dataloader
    patch_data = True
    batch_size = 64
    num_workers = 4
    # Augmentation
    use_augmentation = False
    # Save + log
    save_top_k = 3
    num_log_images = 2
    log_image_interval = 5

    def update_from_args(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)


# ======================== QAT Helpers (FX Graph Mode) ========================

def prepare_qat_model(model, device):
    """Prepare model for QAT using FX Graph Mode."""
    backend = "qnnpack"
    torch.backends.quantized.engine = backend

    qconfig_mapping = build_qat_qconfig_mapping()
    example_inputs = (torch.randn(1, 3, 100, 100),)

    model.train()
    model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    model.to(device)

    print("FX Graph Mode QAT (model): fake-quant observers inserted.")
    return model


def convert_qat_model(model):
    """Convert a QAT-prepared FX model to a fully quantized INT8 model."""
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        model = quantize_fx.convert_fx(model)
    return model


# ======================== LightningModule ========================

class DPEDLightningModule(pl.LightningModule):
    def __init__(self, model, config):
        super(DPEDLightningModule, self).__init__()
        self.model = model
        self.config = config

        # GPU augmentation (applied in on_after_batch_transfer, training only)
        self.augmentation = GPUAugmentation() if getattr(config, "use_augmentation", False) == "True" else None

        self.criterion = build_criterion(config.loss_version, config)
        self.loss_component_names = LOSS_REGISTRY[config.loss_version]["component_names"]

        for param in self.criterion.parameters():
            param.requires_grad = False
        self.register_buffer("max_psnr_value", torch.tensor(100.0))
        self.register_buffer("max_pixel_value", torch.tensor(1.0))
        self.register_buffer("zero", torch.tensor(0.0))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        inputs, targets = batch
        if self.augmentation is not None and self.training:
            inputs, targets = self.augmentation(inputs, targets)
        return inputs, targets

    def forward(self, x):
        return self.model(x)

    def _log_loss_components(self, prefix, loss, components):
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        for name, value in zip(self.loss_component_names, components):
            self.log(f"{prefix}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, _):
        inputs, targets = batch
        outputs = self(inputs)
        outputs = outputs + torch.randn_like(outputs) * 1e-6
        loss, *components = self.criterion(outputs, targets)
        self._log_loss_components("train", loss, components)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss, *components = self.criterion(outputs, targets)
        self._log_loss_components("val", loss, components)

        mse = ((outputs - targets) ** 2).mean(dim=(1, 2, 3))
        is_mse_zero = torch.isclose(mse, self.zero, rtol=1e-05, atol=1e-06)
        psnr = torch.where(is_mse_zero, self.max_psnr_value, 10 * torch.log10(self.max_pixel_value**2 / mse))
        psnr = psnr.mean()
        self.log("val/psnr", psnr, sync_dist=True, prog_bar=True)

        if self.config.use_wandb == "True" and batch_idx == 0 and self.current_epoch % self.config.log_image_interval == 0:
            images = []
            for i in range(min(self.config.num_log_images, inputs.shape[0])):
                inp_img = (inputs[i].permute(1, 2, 0).cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
                out_img = (outputs[i].permute(1, 2, 0).cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
                tgt_img = (targets[i].permute(1, 2, 0).cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
                images.append(wandb.Image(inp_img, caption=f"Input {i}"))
                images.append(wandb.Image(out_img, caption=f"Output {i}"))
                images.append(wandb.Image(tgt_img, caption=f"Target {i}"))
            self.logger.experiment.log({"val_samples": images})
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = build_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ======================== Wandb helpers ========================

def build_wandb_config(config):
    loss_info = LOSS_REGISTRY[config.loss_version]
    wandb_cfg = {
        "model_name": config.model_name,
        "channels": config.channels,
        "task": "dped",
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "scheduler": SCHEDULER_REGISTRY.get(config.scheduler_type, config.scheduler_type),
        "eta_min": config.eta_min,
        "loss_version": config.loss_version,
        "loss": loss_info["description"],
        "limit_train_batches": config.limit_train_batches,
        "limit_val_batches": config.limit_val_batches,
    }

    if config.scheduler_type == "cosine_warm_restarts":
        wandb_cfg.update({"T_0": config.T_0, "T_mult": config.T_mult})
    elif config.scheduler_type == "cosine_warmup":
        wandb_cfg.update({"warmup_epochs": config.warmup_epochs, "warmup_start_factor": config.warmup_start_factor})

    for key in loss_info["weight_keys"]:
        wandb_cfg[key] = getattr(config, key)
    return wandb_cfg


# ======================== Main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on the DPED dataset")

    # Paths
    parser.add_argument("--data_dir", type=str, default="../data/dped", help="Path to the DPED dataset")
    parser.add_argument("--ckpt_dir", type=str, default="../../ckpts", help="Directory to save checkpoints")

    # Training
    parser.add_argument("--patch_data", type=str, choices=["True", "False"], default="True", help="Use patch data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2, help="Accumulate gradients over N batches")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Max norm for gradient clipping (0 = no clipping)")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm", help="Gradient clipping algorithm ('norm' or 'value')")
    parser.add_argument("--limit_train_batches", type=float, default=0.1, help="Fraction of training batches per epoch")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Fraction of validation batches per epoch")

    # Optimizer / Scheduler
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--scheduler_type", type=str, default="cosine_warm_restarts", choices=list(SCHEDULER_REGISTRY), help="LR scheduler type")
    parser.add_argument("--T_0", type=int, default=10, help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--T_mult", type=int, default=2, help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--eta_min", type=float, default=5e-6, help="Minimum LR for cosine schedulers")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs (cosine_warmup only)")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1, help="LR start factor for warmup")

    # Model (model only)
    parser.add_argument("--model_name", type=str, default="modelv07_loss01", help="Experiment name for logging and checkpointing")
    parser.add_argument("--channels", type=int, default=24, help="Number of channels in model")
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
        help="Model variant used for ablations",
    )
    # Loss
    parser.add_argument("--precision", type=str, default="32", choices=["16-mixed", "bf16", "32"], help="Precision for training")
    parser.add_argument("--loss_version", type=int, default=1, choices=[1, 2, 3], help="Loss version")
    parser.add_argument("--weight_psnr", type=float, default=2.0, help="Weight for PSNR loss")
    parser.add_argument("--weight_msmse", type=float, default=1.0, help="Weight for MSMSE loss (loss_version=3)")
    parser.add_argument("--weight_msssim", type=float, default=100.0, help="Weight for MSSSIM loss (loss_version=1)")
    parser.add_argument("--weight_cosine", type=float, default=1.0, help="Weight for cosine similarity loss (loss_version=2)")
    parser.add_argument("--weight_outlier", type=float, default=1.0, help="Weight for outlier-aware loss")
    parser.add_argument("--weight_perceptual", type=float, default=10.0, help="Weight for perceptual loss")

    # Logging
    parser.add_argument("--use_wandb", type=str, choices=["True", "False"], default="False", help="Enable Wandb logging")
    parser.add_argument("--log_image_interval", type=int, default=5, help="Log sample images every N epochs")
    parser.add_argument("--num_log_images", type=int, default=3, help="Number of sample images to log")

    # Augmentation
    parser.add_argument("--use_augmentation", type=str, choices=["True", "False"], default="False", help="Enable GPU data augmentation")

    # QAT
    parser.add_argument("--qat", type=str, choices=["True", "False"], default="False", help="Enable quantization-aware training")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to pretrained checkpoint to load before QAT")
    parser.add_argument(
        "--save_int8_path",
        type=str,
        default=None,
        help="Compatibility arg for ablation scripts; unused in this trainer.",
    )

    # Resume
    parser.add_argument("--ckpt-path", type=str, default=None, help="Continue training from checkpoint")

    args = parser.parse_args()

    config = Config()
    config.update_from_args(args)

    data_path_list = [
        ("iphone", Path(config.data_dir) / "iphone"),
    ]
    datamodule = DPEDDataModule(data_dir=data_path_list, patch_data=config.patch_data, batch_size=config.batch_size, num_workers=config.num_workers)

    model = build_model(config.channels, model_name=config.ablation_model)

    if args.pretrained_ckpt:
        print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        dped_module = DPEDLightningModule.load_from_checkpoint(args.pretrained_ckpt, model=model, config=config)
        model = dped_module.model

    if args.qat == "True":
        print("Preparing model for quantization-aware training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qat_model = prepare_qat_model(model, device)
        dped_module = DPEDLightningModule(qat_model, config=config)
        config.precision = "32"
        print("QAT preparation complete.")
    else:
        dped_module = DPEDLightningModule(model, config=config)

    total_params = sum(p.numel() for p in dped_module.model.parameters())
    trainable_params = sum(p.numel() for p in dped_module.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    loss_checkpoint = ModelCheckpoint(
        monitor="val/psnr",
        dirpath=config.ckpt_dir,
        filename=f"{config.model_name}_epoch{{epoch:02d}}_psnr{{val/psnr:.2f}}",
        save_top_k=config.save_top_k,
        auto_insert_metric_name=False,
        mode="max",
    )

    wandb_logger = None
    if config.use_wandb == "True":
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb_logger = WandbLogger(project="MobileIE-dped", name=config.model_name, log_model=False)
        wandb_logger.experiment.config.update(build_wandb_config(config))
        wandb_logger.watch(model, log="gradients", log_freq=500)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu",
        devices=1,
        precision=getattr(config, "precision", "32"),
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[lr_monitor, loss_checkpoint],
        default_root_dir=config.ckpt_dir,
        logger=wandb_logger,
        enable_progress_bar=(config.use_wandb != "True"),
        enable_model_summary=True,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
    )

    trainer_fit_args = {
        "model": dped_module,
        "datamodule": datamodule,
    }
    if args.ckpt_path:
        trainer_fit_args.update({"ckpt_path": args.ckpt_path})
        print(f"Continue from ckpt: {args.ckpt_path=}")

    trainer.fit(**trainer_fit_args)
