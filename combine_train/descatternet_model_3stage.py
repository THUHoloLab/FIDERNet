
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from arch_util import SpectralGate2D


class IdentityGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batchnorm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batchnorm)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DescatterNetUNet(nn.Module):
    """
    U-Net style DescatterNet backbone with optional SpectralGate2D at bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.1,
        batchnorm: bool = True,
        use_spectral_gate: bool = False,
        gate_a_px_input: float = 8.0,
        gate_bandwidth: float = 0.06,
        gate_harmonics: Tuple[int, ...] = (1, 2),
        gate_alpha_max: float = 0.7,
        gate_init_alpha: float = 0.0,
        gate_per_channel: bool = True,
    ):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c, batchnorm=batchnorm)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(dropout * 0.5)

        self.enc2 = ConvBlock(c, c * 2, batchnorm=batchnorm)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(dropout)

        self.enc3 = ConvBlock(c * 2, c * 4, batchnorm=batchnorm)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(dropout)

        self.enc4 = ConvBlock(c * 4, c * 8, batchnorm=batchnorm)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(dropout)

        bottleneck_channels = c * 16
        self.bottleneck = ConvBlock(c * 8, bottleneck_channels, batchnorm=batchnorm)

        if use_spectral_gate:
            down_scale = 2 ** 4
            self.spectral_gate = SpectralGate2D(
                channels=bottleneck_channels,
                a_px=gate_a_px_input / float(down_scale),
                bandwidth=gate_bandwidth,
                harmonics=tuple(int(v) for v in gate_harmonics),
                alpha_max=gate_alpha_max,
                init_alpha=gate_init_alpha,
                per_channel=gate_per_channel,
            )
        else:
            self.spectral_gate = IdentityGate()

        self.up4 = nn.ConvTranspose2d(bottleneck_channels, c * 8, kernel_size=2, stride=2)
        self.dec4_drop = nn.Dropout2d(dropout)
        self.dec4 = ConvBlock(c * 16, c * 8, batchnorm=batchnorm)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3_drop = nn.Dropout2d(dropout)
        self.dec3 = ConvBlock(c * 8, c * 4, batchnorm=batchnorm)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2_drop = nn.Dropout2d(dropout)
        self.dec2 = ConvBlock(c * 4, c * 2, batchnorm=batchnorm)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1_drop = nn.Dropout2d(dropout)
        self.dec1 = ConvBlock(c * 2, c, batchnorm=batchnorm)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.drop1(self.pool1(e1))

        e2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(e2))

        e3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(e3))

        e4 = self.enc4(p3)
        p4 = self.drop4(self.pool4(e4))

        b = self.bottleneck(p4)
        b = self.spectral_gate(b)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4_drop(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_drop(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_drop(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_drop(d1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return self.out_act(out)


def build_descatternet(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 16,
    dropout: float = 0.1,
    batchnorm: bool = True,
    use_spectral_gate: bool = False,
    gate_a_px_input: float = 4.0,
    gate_bandwidth: float = 0.06,
    gate_harmonics: Tuple[int, ...] = (1, 2),
    gate_alpha_max: float = 0.7,
    gate_init_alpha: float = 0.0,
    gate_per_channel: bool = True,
) -> DescatterNetUNet:
    return DescatterNetUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        dropout=dropout,
        batchnorm=batchnorm,
        use_spectral_gate=use_spectral_gate,
        gate_a_px_input=gate_a_px_input,
        gate_bandwidth=gate_bandwidth,
        gate_harmonics=gate_harmonics,
        gate_alpha_max=gate_alpha_max,
        gate_init_alpha=gate_init_alpha,
        gate_per_channel=gate_per_channel,
    )


if __name__ == "__main__":
    x = torch.randn(1, 1, 448, 448)
    model_plain = build_descatternet()
    model_gate = build_descatternet(use_spectral_gate=True)
    y1 = model_plain(x)
    y2 = model_gate(x)
    print("plain:", tuple(y1.shape))
    print("gate :", tuple(y2.shape))
