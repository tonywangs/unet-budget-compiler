"""Generated standalone U-Net. Outputs raw NCHW logits; requires PyTorch.
Compiler 0.1.0; normalized specification SHA-256: de34da55ea3a9de87d972c339e5a1e41134a995d032dc2e84691cb49246f0660
Do not edit if you need byte-for-byte regeneration."""
import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    """Padded double convolutions, floor pooling, nearest resize, skip concatenation."""
    def __init__(self):
        super().__init__()
        self.enc_0_a = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True)
        self.enc_0_b = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
        self.enc_1_a = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=True)
        self.enc_1_b = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=True)
        self.enc_2_a = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        self.enc_2_b = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.dec_1_a = nn.Conv2d(24, 8, kernel_size=3, padding=1, bias=True)
        self.dec_1_b = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=True)
        self.dec_0_a = nn.Conv2d(12, 4, kernel_size=3, padding=1, bias=True)
        self.dec_0_b = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
        self.head = nn.Conv2d(4, 2, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("expected NCHW input with 1 channels")
        if x.shape[0] < 1 or min(x.shape[-2:]) < 4:
            raise ValueError("batch must be nonempty and height/width must be >= 4")
        x = F.relu(self.enc_0_a(x))
        x = F.relu(self.enc_0_b(x))
        skip_0 = x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.enc_1_a(x))
        x = F.relu(self.enc_1_b(x))
        skip_1 = x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.enc_2_a(x))
        x = F.relu(self.enc_2_b(x))
        x = F.interpolate(x, size=skip_1.shape[-2:], mode="nearest")
        x = torch.cat((skip_1, x), dim=1)
        x = F.relu(self.dec_1_a(x))
        x = F.relu(self.dec_1_b(x))
        x = F.interpolate(x, size=skip_0.shape[-2:], mode="nearest")
        x = torch.cat((skip_0, x), dim=1)
        x = F.relu(self.dec_0_a(x))
        x = F.relu(self.dec_0_b(x))
        return self.head(x)
