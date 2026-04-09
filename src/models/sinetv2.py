"""
sinetv2.py — SINetV2 model for Experiment 1 (ACD1K only)
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Yansong Jia (Experiment 1 Lead)

Lightweight encoder–decoder architecture inspired by SINetV2 for camouflaged
object segmentation. Designed for binary segmentation of military camouflage
targets on ACD1K with an input resolution of 352×352, but supports arbitrary
spatial sizes (multiples of 32 recommended).

API:
    model = SINetV2(in_channels=3, base_channels=32)
    logits = model(images)  # [B, 1, H, W]
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Standard 2D conv → batch norm → ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution used in encoder/decoder blocks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class EncoderBlock(nn.Module):
    """Encoder stage: depthwise separable conv + channel attention + downsample."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride=1)
        self.attn = ChannelAttention(out_channels)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.conv(x)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder stage: bilinear upsample + fusion with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SINetV2(nn.Module):
    """
    Simplified SINetV2-style encoder–decoder for binary camouflage segmentation.

    Args:
        in_channels: Number of input image channels (default: 3).
        base_channels: Base number of feature channels (default: 32).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Encoder: four stages, progressively downsample by 2×.
        enc_channels: List[int] = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.stem = ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=1)

        self.enc1 = EncoderBlock(base_channels, enc_channels[0])
        self.enc2 = EncoderBlock(enc_channels[0], enc_channels[1])
        self.enc3 = EncoderBlock(enc_channels[1], enc_channels[2])
        self.enc4 = EncoderBlock(enc_channels[2], enc_channels[3])

        # Bottleneck
        self.bottleneck = ConvBNReLU(enc_channels[3], enc_channels[3], kernel_size=3, stride=1)

        # Decoder: symmetric to encoder with skip connections.
        self.dec4 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2])
        self.dec3 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1])
        self.dec2 = DecoderBlock(enc_channels[1], enc_channels[0], enc_channels[0])
        self.dec1 = DecoderBlock(enc_channels[0], base_channels, base_channels)

        # Final prediction head: single-channel logits for BCEWithLogitsLoss.
        self.pred = nn.Conv2d(base_channels, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Logits tensor of shape [B, 1, H, W].
        """
        stem = self.stem(x)          # [B, C, H, W]

        x, s1 = self.enc1(stem)      # 1/2
        x, s2 = self.enc2(x)         # 1/4
        x, s3 = self.enc3(x)         # 1/8
        x, _  = self.enc4(x)         # 1/16

        x = self.bottleneck(x)

        # Decoder uses highest-level skip from s3, then s2, s1, and stem.
        x = self.dec4(x, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, stem)

        logits = self.pred(x)
        return logits


def build_sinetv2(in_channels: int = 3, base_channels: int = 32) -> SINetV2:
    """
    Convenience factory to build SINetV2, mirroring build_model patterns
    from other experiments.
    """
    return SINetV2(in_channels=in_channels, base_channels=base_channels)

