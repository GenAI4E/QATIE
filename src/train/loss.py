import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class PSNRLoss(nn.Module):
    """
    PSNR Loss: Computes a differentiable loss based on Peak Signal-to-Noise Ratio.
    
    ONNX-friendly implementation with proper buffer registration and no dynamic control flow.
    
    Formula:
        RMSE = sqrt(mean((pred - target)^2))
        PSNR = 20 * log10(max_pixel_value / RMSE)
        Loss = (50.0 - PSNR) / 100.0  # Normalized to [0, 1] range
    
    Args:
        reduction: Reduction method, must be 'mean' (default: 'mean')
        toY: If True, converts RGB to Y channel (luminance) using ITU-R BT.601 coefficients
             Coefficients: Y = 0.257*R + 0.504*G + 0.098*B + 16
    """
    def __init__(self, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        if reduction != 'mean':
            raise ValueError(f"Only 'mean' reduction is supported, got '{reduction}'")
        self.toY = toY
        
        # Register RGB to Y conversion coefficients as buffer for ONNX compatibility
        # ITU-R BT.601 coefficients: Y = 0.257*R + 0.504*G + 0.098*B + 16
        # Scaled by 255: [65.481, 128.553, 24.966]
        if self.toY:
            rgb_to_y_coef = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.register_buffer('rgb_to_y_coef', rgb_to_y_coef)
        
        # Register constants as buffers for ONNX tracing
        self.register_buffer('eps', torch.tensor(1e-8))
        self.register_buffer('normalization_offset', torch.tensor(16.0))
        self.register_buffer('normalization_scale', torch.tensor(255.0))

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image tensor of shape (B, C, H, W)
            target: Target image tensor of shape (B, C, H, W)
        
        Returns:
            loss: Scalar tensor representing the PSNR-based loss
        """
        # Convert RGB to Y channel (luminance) if specified
        if self.toY:
            # Apply ITU-R BT.601 conversion: Y = 0.257*R + 0.504*G + 0.098*B + 16
            pred = (pred * self.rgb_to_y_coef).sum(dim=1, keepdim=True) + self.normalization_offset
            target = (target * self.rgb_to_y_coef).sum(dim=1, keepdim=True) + self.normalization_offset
            
            # Normalize to [0, 1] range
            pred = pred / self.normalization_scale
            target = target / self.normalization_scale
        
        # Compute pixel-wise difference
        diff = pred - target
        
        # Compute RMSE per image in batch
        # Shape: (B,) after mean over (C, H, W) dimensions
        mse_per_image = (diff ** 2).mean(dim=(1, 2, 3))
        rmse_per_image = torch.sqrt(mse_per_image + self.eps)
        
        # Compute PSNR in dB scale
        # Max pixel value is 1.0 (normalized images)
        psnr_per_image = 20 * torch.log10(1.0 / rmse_per_image)
        psnr_mean = psnr_per_image.mean()
        
        # Convert PSNR to loss: normalize to [0, 1] range
        # Higher PSNR = better quality = lower loss
        # Typical PSNR range: 20-50 dB, so (50 - PSNR) / 100 maps to [0, 0.3]
        loss = (50.0 - psnr_mean) / 100.0
        
        return loss
    
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.window = None
        self.size_average = True
        self.value_range = None
    
    def forward(self, pred, target):
        with torch.amp.autocast('cuda', enabled=False):
            return 1 - self.ssim(pred.float(), target.float())

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, self.sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, pred, target):
        if self.value_range is not None:
            L = self.value_range
        else:
            if torch.max(pred) > 128:
                max_val = 255
            else:
                max_val = 1
            if torch.min(pred) < 0:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val

        pad = 0
        (_, channel, height, width) = pred.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            self.window = self.create_window(real_size, channel=channel).to(pred.device)
        
        mu1 = F.conv2d(pred, self.window, padding=pad, groups=channel)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=pad, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        # cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class ContentLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(ContentLoss, self).__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).to('cuda').eval()
        # Extract feature blocks according to the paper: layers 1, 3, 5
        # conv1_2 (layer 1): features[:4]
        # conv2_2 (layer 3): features[:9]  
        # conv3_2 (layer 5): features[:14]
        blocks = []
        blocks.append(vgg.features[:4])
        blocks.append(vgg.features[:9])
        blocks.append(vgg.features[:14])
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, pred, target):
        # is_on_cuda = next(vgg.parameters()).is_cuda
        # if not is_on_cuda:
        #     vgg = vgg.to(pred.device)
        self.mean = self.mean.to(pred.device)
        self.std = self.std.to(pred.device)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            pred = self.transform(pred, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        for j, block in enumerate(self.blocks):
            feat_pred = block(pred)
            feat_target = block(target)
            C_j = feat_pred.shape[1]
            H_j = feat_pred.shape[2]
            W_j = feat_pred.shape[3]
            loss += torch.nn.functional.l1_loss(feat_pred, feat_target) / (C_j * H_j * W_j)
        
        return loss
    
class OutlierAwareLoss(nn.Module):
    def __init__(self,):
        super(OutlierAwareLoss, self).__init__()

    def forward(self, out, lab):
        delta = out - lab
        var = delta.std((2, 3), keepdims=True) / (2 ** .5)
        avg = delta.mean((2, 3), True)
        weight = torch.tanh((delta - avg).abs() / (var + 1e-6)).detach()       
        loss = (delta.abs() * weight).mean()
        return loss

class MSMSE(torch.nn.Module):
    def __init__(self, ms_weights=[1.0, 1.0, 1.0]):
        super(MSMSE, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.half_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ms_weights = ms_weights

    def forward(self, pred, target):
        half_inp = self.half_pooling(pred)
        quarter_inp = self.half_pooling(half_inp)
        half_tgt = self.half_pooling(target)
        quarter_tgt = self.half_pooling(half_tgt)
        loss = self.mse(pred, target) * self.ms_weights[0]
        loss += self.mse(half_inp, half_tgt) * self.ms_weights[1]
        loss += self.mse(quarter_inp, quarter_tgt) * self.ms_weights[2]
        return loss

# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================

class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_psnr: float=1.0, weight_msssim: float=500.0, weight_outlier: float=1.0, resize=True, return_loss_components=False):
        super(CombinedLoss, self).__init__()
        self.weight_psnr = weight_psnr
        self.weight_msssim = weight_msssim
        self.weight_outlier = weight_outlier
        self.return_loss_components = return_loss_components

        self.psnr = PSNRLoss(toY=True)
        self.msssim = MSSSIM()
        self.outlier = OutlierAwareLoss()

    def forward(self, pred, target):
        psnr_loss = self.psnr(pred, target)
        msssim_loss = self.msssim(pred, target)
        outlier_loss = self.outlier(pred, target)
        loss = self.weight_psnr * psnr_loss \
            + self.weight_msssim * msssim_loss \
            + self.weight_outlier * outlier_loss
        if self.return_loss_components:
            return loss, psnr_loss, msssim_loss, outlier_loss
        return loss


# Outlier , PSNR , CosineSim
class CombinedLossV2(torch.nn.Module):
    def __init__(self, weight_psnr: float=1.0, weight_cosine: float=1.0, weight_outlier: float=1.0, resize=True, return_loss_components=False):
        super(CombinedLossV2, self).__init__()
        self.weight_psnr = weight_psnr
        self.weight_cosine = weight_cosine
        self.weight_outlier = weight_outlier
        self.return_loss_components = return_loss_components

        self.psnr = PSNRLoss(toY=True)
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.outlier = OutlierAwareLoss()

    def forward(self, pred, target):
        psnr_loss = self.psnr(pred, target)
        cosine_loss = 1 - self.cosine(pred.reshape(pred.size(0), -1), target.reshape(target.size(0), -1)).mean()
        outlier_loss = self.outlier(pred, target)
        loss = self.weight_psnr * psnr_loss
        loss += self.weight_cosine * cosine_loss
        loss += self.weight_outlier * outlier_loss
        if self.return_loss_components:
            return loss, psnr_loss, cosine_loss, outlier_loss
        return loss


# MSMSE, MSSSIM, Perceptual
class CombinedLossV3(torch.nn.Module):
    def __init__(self, weight_msssim: float=1.0, weight_msmse: float=1.0, weight_perceptual: float=500.0, resize=True, return_loss_components=False):
        super(CombinedLossV3, self).__init__()
        self.weight_msssim = weight_msssim
        self.weight_msmse = weight_msmse
        self.weight_perceptual = weight_perceptual
        self.return_loss_components = return_loss_components

        self.msssim = MSSSIM()
        self.msmse = MSMSE()
        self.perceptual = ContentLoss(resize=resize)

    def forward(self, pred, target):
        msssim_loss = self.msssim(pred, target)
        msmse_loss = self.msmse(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        loss = self.weight_msssim * msssim_loss
        loss += self.weight_msmse * msmse_loss
        loss += self.weight_perceptual * perceptual_loss
        if self.return_loss_components:
            return loss, msmse_loss, msssim_loss, perceptual_loss
        return loss