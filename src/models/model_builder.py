"""
model_builder.py — single source of truth for model construction and checkpoint loading.

Imported by train_qat.py, to_tflite.py, and any future scripts.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional


class ConcatFP32(nn.Module):
    """
    FP32 concat wrapper.

    In FX/QAT INT8 graphs, a quantized `cat` op requires identical quantization
    parameters across inputs. By running concat in FP32 (qconfig=None), we avoid
    that warning and potential accuracy loss.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, tensors):
        return torch.cat(tensors, dim=self.dim)


def build_qat_qconfig_mapping(concat_fp32_in_fp32: bool = True):
    """FX QAT QConfigMapping (qnnpack) for HybridMixUNet — shared by train_qat.py and quantize.py.

    ``head`` stays FP32 (residual output).

    If ``concat_fp32_in_fp32`` is True (default), ``ConcatFP32`` runs in FP32 so FX does not
    emit quantized ``cat`` with mismatched input qparams. Use False only to match older QAT
    checkpoints trained before that rule existed (same graph / state dict keys as training).
    """
    from torch.ao.quantization import get_default_qat_qconfig_mapping

    backend = "qnnpack"
    qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    qconfig_mapping.set_module_name("head", qconfig=None)
    if concat_fp32_in_fp32:
        qconfig_mapping.set_object_type(ConcatFP32, qconfig=None)
    return qconfig_mapping


# -------------------------
# Baseline blocks (reuse)
# -------------------------
class DownBlock(nn.Module):
    """2-branch stride-2 conv + tanh gating.
    Returns: (left, mid=gated, right)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.a = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=True)
        self.b = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=True)
        self.skip_mul = FloatFunctional()

    def forward(self, x):
        xa = torch.tanh(self.a(x))
        xb = torch.tanh(self.b(x))
        return xa, self.skip_mul.mul(xa, xb), xb


class UpBlock(nn.Module):
    """Upsample + conv. Uses target spatial size for shape safety."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)

    def forward(self, x, size_hw):
        x = F.interpolate(x, size=size_hw, mode="nearest")
        return self.conv(x)


class Conv3x3Refiner(nn.Module):
    """Simple 3x3 conv + nonlinearity used for ablation baselines."""

    def __init__(self, in_ch: int, out_ch: int, relu_slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(relu_slope, inplace=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.act(self.conv(x))


# -------------------------
# UNet conv block 
# -------------------------
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_size, in_size, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(in_size, affine=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.concat_fp32 = ConcatFP32(dim=1)

        self.conv_2 = nn.Conv2d(in_size * 2, out_size, 3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_3 = nn.Conv2d(out_size, out_size, 3, padding=1, bias=True)
        self.relu_3 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_1_1_1 = nn.Conv2d(in_size, in_size, 1, 1, 0)
        self.conv_1_1 = nn.Conv2d(in_size * 2, out_size, 1, 1, 0)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.norm1(out)
        out = self.relu_1(out)

        out1 = self.conv_1_1_1(x)
        out2 = self.concat_fp32([out, out1])

        out = self.conv_2(out2)
        out = self.relu_2(out)
        out = self.conv_3(out)
        out = self.relu_3(out)

        res = self.conv_1_1(out2)
        out = out + res
        return out


# -------------------------
# Hybrid model (Option B - interleaving per diagram)
# Scales: S/2 -> S/4 -> S/8 bottleneck -> Up1(S/4) -> Up2(S/2) -> full
# Uses baseline DownBlocks + UNetConvBlock refiners
# Output: y = clamp(x + delta, 0, 1)
# -------------------------
class HybridMixUNet(nn.Module):
    def __init__(self, c=24, relu_slope=0.2):
        super().__init__()
        self.c = c
        self.concat_fp32 = ConcatFP32(dim=1)

        # Down path (baseline)
        self.down1 = DownBlock(3, c)        # S/2, ch=c
        self.down2 = DownBlock(c, 2 * c)    # S/4, ch=2c

        # UNet-style refiners
        self.unet_s2 = UNetConvBlock(c, c, relu_slope=relu_slope)                # S/2
        self.unet_s4 = UNetConvBlock(2 * c, 2 * c, relu_slope=relu_slope)        # S/4

        # Bottleneck at S/8
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = UNetConvBlock(2 * c, 4 * c, relu_slope=relu_slope)     # S/8, ch=4c

        # Up path
        self.up_to_s4 = UpBlock(4 * c, 2 * c)   # S/8 -> S/4
        self.up_to_s2 = UpBlock(2 * c, c)       # S/4 -> S/2
        self.up_to_s1 = UpBlock(c, c)           # S/2 -> S/1

        # Fusion (pre-reduce then UNetConv)
        # At S/4 fuse: [up(S/4)=2c, unet_s4=2c, skip2(cat lf2,rt2)=4c] => 8c -> 2c
        self.fuse4_reduce = nn.Conv2d(8 * c, 2 * c, 1, 1, 0)
        self.fuse4_refine = UNetConvBlock(2 * c, 2 * c, relu_slope=relu_slope)

        # At S/2 fuse: [up(S/2)=c, unet_s2=c, skip1(cat lf1,rt1)=2c] => 4c -> c
        self.fuse2_reduce = nn.Conv2d(4 * c, c, 1, 1, 0)
        self.fuse2_refine = UNetConvBlock(c, c, relu_slope=relu_slope)

        # Head: predict residual (delta)
        self.head = nn.Conv2d(c, 3, 3, 1, 1, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # ----- Down 1 (S/2) -----
        lf1, mid1, rt1 = self.down1(x)  # all: (B,c,H/2,W/2)
        s2 = self.unet_s2(mid1)         # (B,c,H/2,W/2)
        skip1 = self.concat_fp32([lf1, rt1])  # (B,2c,H/2,W/2)

        # ----- Down 2 (S/4) -----
        lf2, mid2, rt2 = self.down2(s2)       # all: (B,2c,H/4,W/4)
        s4 = self.unet_s4(mid2)               # (B,2c,H/4,W/4)
        skip2 = self.concat_fp32([lf2, rt2])  # (B,4c,H/4,W/4)

        # ----- Bottleneck (S/8) -----
        b = self.pool(s4)                      # (B,2c,H/8,W/8)
        b = self.bottleneck(b)                 # (B,4c,H/8,W/8)

        # ----- Up to S/4 + fuse (Up1 ) -----
        b_up4 = self.up_to_s4(b, size_hw=s4.shape[-2:])  # (B,2c,H/4,W/4)
        fuse4 = self.concat_fp32([b_up4, s4, skip2])     # (B,8c,H/4,W/4)
        fuse4 = self.fuse4_reduce(fuse4)                 # (B,2c,H/4,W/4)
        fuse4 = self.fuse4_refine(fuse4)                 # (B,2c,H/4,W/4)

        # ----- Up to S/2 + fuse (Up2 ) -----
        up2 = self.up_to_s2(fuse4, size_hw=s2.shape[-2:])   # (B,c,H/2,W/2)
        fuse2 = self.concat_fp32([up2, s2, skip1])          # (B,4c,H/2,W/2)
        fuse2 = self.fuse2_reduce(fuse2)                    # (B,c,H/2,W/2)
        fuse2 = self.fuse2_refine(fuse2)                    # (B,c,H/2,W/2)

        # ----- Up to full -----
        up1 = self.up_to_s1(fuse2, size_hw=x.shape[-2:])    # (B,c,H,W)

        delta = self.head(up1)                              # (B,3,H,W)
        y = torch.clamp(x + delta, 0.0, 1.0)
        return y


class HybridMixUNetAblation(HybridMixUNet):
    """HybridMixUNet component ablations (high-signal, comparable variants)."""

    def __init__(
        self,
        c: int = 24,
        relu_slope: float = 0.2,
        ablate_fuse_refine: bool = False,
        ablate_resolution_refine: bool = False,
        ablate_residual_skip: bool = False,
    ):
        super().__init__(c=c, relu_slope=relu_slope)

        self.ablate_residual_skip = bool(ablate_residual_skip)

        # Ablate UNet-style refiners at S/2 and S/4 with simple 3x3 convs.
        if ablate_resolution_refine:
            self.unet_s2 = Conv3x3Refiner(c, c, relu_slope=relu_slope)
            self.unet_s4 = Conv3x3Refiner(2 * c, 2 * c, relu_slope=relu_slope)

        # Ablate fusion refinement blocks (replace UNetConvBlock with 3x3 conv).
        if ablate_fuse_refine:
            self.fuse4_refine = Conv3x3Refiner(2 * c, 2 * c, relu_slope=relu_slope)
            self.fuse2_refine = Conv3x3Refiner(c, c, relu_slope=relu_slope)

    def forward(self, x):
        # Same forward as HybridMixUNet, except output residual formulation.
        lf1, mid1, rt1 = self.down1(x)  # all: (B,c,H/2,W/2)
        s2 = self.unet_s2(mid1)  # (B,c,H/2,W/2)
        skip1 = self.concat_fp32([lf1, rt1])  # (B,2c,H/2,W/2)

        lf2, mid2, rt2 = self.down2(s2)  # all: (B,2c,H/4,W/4)
        s4 = self.unet_s4(mid2)  # (B,2c,H/4,W/4)
        skip2 = self.concat_fp32([lf2, rt2])  # (B,4c,H/4,W/4)

        b = self.pool(s4)  # (B,2c,H/8,W/8)
        b = self.bottleneck(b)  # (B,4c,H/8,W/8)

        b_up4 = self.up_to_s4(b, size_hw=s4.shape[-2:])  # (B,2c,H/4,W/4)
        fuse4 = self.concat_fp32([b_up4, s4, skip2])  # (B,8c,H/4,W/4)
        fuse4 = self.fuse4_reduce(fuse4)  # (B,2c,H/4,W/4)
        fuse4 = self.fuse4_refine(fuse4)  # (B,2c,H/4,W/4)

        up2 = self.up_to_s2(fuse4, size_hw=s2.shape[-2:])  # (B,c,H/2,W/2)
        fuse2 = self.concat_fp32([up2, s2, skip1])  # (B,4c,H/2,W/2)
        fuse2 = self.fuse2_reduce(fuse2)  # (B,c,H/2,W/2)
        fuse2 = self.fuse2_refine(fuse2)  # (B,c,H/2,W/2)

        up1 = self.up_to_s1(fuse2, size_hw=x.shape[-2:])  # (B,c,H,W)
        delta = self.head(up1)  # (B,3,H,W)

        if self.ablate_residual_skip:
            y = torch.clamp(delta, 0.0, 1.0)
        else:
            y = torch.clamp(x + delta, 0.0, 1.0)
        return y


def build_model(channels: int, model_name: str = "hybrid_base"):
    """Build ablation-ready model variants.

    Args:
        channels: base channel count.
        model_name: one of
            [hybrid_base,
             hybrid_ablate_fuse_refine, hybrid_ablate_resolution_refine, hybrid_ablate_residual,
             hybrid_ablate_all]
    """
    model_name = (model_name or "hybrid_base").lower()
    if model_name == "hybrid_base":
        return HybridMixUNet(c=channels)
    if model_name == "hybrid_ablate_fuse_refine":
        return HybridMixUNetAblation(c=channels, ablate_fuse_refine=True)
    if model_name == "hybrid_ablate_resolution_refine":
        return HybridMixUNetAblation(c=channels, ablate_resolution_refine=True)
    if model_name == "hybrid_ablate_residual":
        return HybridMixUNetAblation(c=channels, ablate_residual_skip=True)
    if model_name == "hybrid_ablate_all":
        return HybridMixUNetAblation(
            c=channels,
            ablate_fuse_refine=True,
            ablate_resolution_refine=True,
            ablate_residual_skip=True,
        )
    raise ValueError(
        f"Unknown model_name={model_name!r}. "
        f"Choose from supported hybrid variants."
    )


def is_fx_quantized_full_model_checkpoint(ckpt_path: str | Path) -> bool:
    """True if this looks like ``torch.save(model)`` after ``quantize_fx.convert_fx`` (FX GraphModule).

    Those archives are fragile across PyTorch versions and are not supported by
    ``load_checkpoint_weights`` / ONNX export (use a Lightning ``.ckpt`` with ``state_dict``).
    """
    p = Path(ckpt_path)
    if not p.is_file():
        return False
    try:
        with zipfile.ZipFile(p, "r") as zf:
            data_pkls = [n for n in zf.namelist() if n.endswith("data.pkl")]
            for n in data_pkls:
                with zf.open(n) as f:
                    head = f.read(32768)
                # Serialized GraphModule from convert_fx; Lightning ckpt pickles rarely contain this.
                if b"reduce_graph_module" in head and (
                    b"torch.fx" in head or b"ctorch.fx" in head or b"graph_module" in head
                ):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def load_torch_checkpoint_trusted(ckpt_path: str | Path):
    """Load a checkpoint root object (dict or state dict). Prefer ``weights_only=True`` when possible."""
    path = str(ckpt_path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 2.0
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint_weights(model, ckpt_path):
    """Load weights from a Lightning checkpoint into a bare model."""
    raw = load_torch_checkpoint_trusted(ckpt_path)
    checkpoint = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dict checkpoint in {ckpt_path!r}, got {type(checkpoint).__name__}")
    # Remove Lightning buffers
    for key in ['max_psnr_value', 'max_pixel_value', 'zero']:
        checkpoint.pop(key, None)
    # Strip 'model.' prefix (Lightning module) and optionally 'inner_model.' (QAT wrapper)
    # Also drop fake-quant observer keys (weight_fake_quant.*, activation_post_process.*)
    state_dict = {}
    for k, v in checkpoint.items():
        if not k.startswith("model."):
            continue
        k = k[len("model."):]
        if k.startswith("inner_model."):
            k = k[len("inner_model."):]
        # Skip fake-quant observer state (not part of the actual model weights)
        if "weight_fake_quant" in k or "activation_post_process" in k:
            continue
        state_dict[k] = v
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        # Identify which missing keys belong to BatchNorm modules
        # (expected for QAT/fused checkpoints where Conv+BN are merged)
        bn_module_names = {
            name for name, mod in model.named_modules()
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        }
        bn_missing = []
        other_missing = []
        for k in incompatible.missing_keys:
            parent = k.rsplit('.', 1)[0] if '.' in k else ''
            if parent in bn_module_names:
                bn_missing.append(k)
            else:
                other_missing.append(k)
        if bn_missing:
            print(f"[load_checkpoint_weights] {len(bn_missing)} BatchNorm key(s) missing "
                  f"(expected for QAT/fused checkpoints, using default identity init)")
        if other_missing:
            print(f"[load_checkpoint_weights] Checkpoint is missing {len(other_missing)} key(s) "
                  f"(keeping default init): {other_missing[:6]}{'...' if len(other_missing) > 6 else ''}")
    if incompatible.unexpected_keys:
        print(f"[load_checkpoint_weights] Checkpoint has {len(incompatible.unexpected_keys)} unexpected key(s) "
              f"(ignored): {incompatible.unexpected_keys[:6]}{'...' if len(incompatible.unexpected_keys) > 6 else ''}")
    return model



# Ordered list of (key_suffix, shape_dim) probes to infer base channel count `c`.
# Each suffix is tried with prefixes "net.", "model.", and "" to handle different
# wrapper patterns (v02: self.net, v06/v04: self.model, v01/v07+: no wrapper).
_CHANNEL_PROBE_KEYS = [
    ("down1.a.weight", 0),             # current HybridMixUNet and ablation variants
    ("ds1.conv1.weight", 0),           # alternative naming
    ("encoder.0.weight", 0),           # fallback
]


def infer_channels_from_checkpoint(ckpt_path: str) -> int:
    """
    Infer the base channel count `c` from a checkpoint's state dict.
    Raises ValueError if no known probe key is found.
    """
    root = load_torch_checkpoint_trusted(ckpt_path)
    raw = root["state_dict"] if isinstance(root, dict) and "state_dict" in root else root
    state_dict = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("inner_model."):
            nk = nk[len("inner_model."):]
        if "weight_fake_quant" in nk or "activation_post_process" in nk:
            continue
        state_dict[nk] = v

    for suffix, dim in _CHANNEL_PROBE_KEYS:
        for prefix in ("net.", "model.", ""):
            key = prefix + suffix
            if key in state_dict:
                c = state_dict[key].shape[dim]
                print(f"[infer_channels] '{key}' shape={tuple(state_dict[key].shape)} → c={c}")
                return c

    raise ValueError(
        f"Could not auto-infer channels from '{ckpt_path}'. "
        f"Please pass --channels explicitly."
    )



# count model params for all models
if __name__ == "__main__":
    for model_name in ["hybrid_base", "hybrid_ablate_fuse_refine", "hybrid_ablate_resolution_refine", "hybrid_ablate_residual", "hybrid_ablate_all"]:
        model = build_model(channels=32, model_name=model_name)
        print(f"Model {model_name} has {sum(p.numel() for p in model.parameters())} parameters")

    for channel in [16, 24, 32, 64]:
        model = build_model(channels=channel, model_name="hybrid_base")
        print(f"Model hybrid_base with channel {channel} has {sum(p.numel() for p in model.parameters())} parameters")