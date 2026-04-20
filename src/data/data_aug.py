import torch
import torch.nn as nn
import kornia.augmentation as K


class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.geometric = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "input"],
        )

        self.photometric = K.AugmentationSequential(
            K.RandomGamma(gamma=(0.85, 1.15), gain=(0.9, 1.1), p=0.5),
            K.ColorJitter(brightness=0.1, contrast=0.1, p=0.3), 
            data_keys=["input"],
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.geometric(inputs, targets)
        inputs = inputs.clamp(0.01, 1.0)
        inputs = self.photometric(inputs)
        inputs = inputs.clamp(0.0, 1.0)
        return inputs, targets
