"""Independent executable specification; does not import the compiler.

Uses module containers and integer-index nearest resize instead of emitted,
unrolled functional calls. Weight copying in tests follows Conv2d traversal.
"""
import torch
from torch import nn


class ReferenceUNet(nn.Module):
    def __init__(self, inputs, classes, depth, width):
        super().__init__()

        def pair(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

        channels = [width * (2**i) for i in range(depth + 1)]
        self.encoder = nn.ModuleList([pair(a, b) for a, b in
                                     zip([inputs] + channels[:-1], channels)])
        self.decoder = nn.ModuleList([pair(channels[i+1] + channels[i], channels[i])
                                     for i in range(depth-1, -1, -1)])
        self.output = nn.Conv2d(width, classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for i, block in enumerate(self.encoder):
            x = block(self.pool(x) if i else x)
            features.append(x)
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            # PyTorch nearest (not nearest-exact): floor(dst_index * src/dst).
            h, w = skip.shape[-2:]
            rows = torch.arange(h, device=x.device) * x.shape[-2] // h
            cols = torch.arange(w, device=x.device) * x.shape[-1] // w
            x = x.index_select(2, rows).index_select(3, cols)
            x = block(torch.cat([skip, x], dim=1))
        return self.output(x)
