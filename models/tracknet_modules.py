"""Building blocks for TrackNet Attention Res-UNet model."""
from typing import Optional
from enum import Enum
import torch
import torch.nn as nn


class InputConvBlock(nn.Module):
    """
    Input convolution block from https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2.

    Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        """
        Input convolution block initializer.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            mid_channels (int, optional): Number of intermediate features. Defaults to number of output features.
        """
        super().__init__()
        if mid_channels is None: mid_channels = out_channels

        self._convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """
        Input convolution block forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Input convolution block output.
        """
        out = self._convs(x)
        return out


class ResBlockType(Enum):
    """Different residual block types."""
    DOWNSAMPLER = 1
    UPSAMPLER = 2
    ENCODER = 3
    DECODER = 4


class ResBlock(nn.Module):
    """
    Residual block for ResNet.

    Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
    """
    def __init__(self, in_channels: int, out_channels: int, type: ResBlockType):
        """
        Residual block initializer.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            type (ResBlockType): Type of residual block.
        """
        super().__init__()

        self._short_cut = lambda x: x
        conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        if type is ResBlockType.DOWNSAMPLER:
            self._short_cut = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        elif type is ResBlockType.UPSAMPLER:
            self._short_cut = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            conv2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2, output_padding=1)
        elif in_channels != out_channels:
            self._short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self._conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self._conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            conv2
        )
        self._conv3 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        """
        Residual block forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Residual block output.
        """
        short_cut = self._short_cut(x)
        x1 = self._conv1(x)
        x2 = self._conv2(x1)
        x3 = self._conv3(x2)
        # print('x1', x1.size())
        # print('x2', x2.size())
        # print('x3', x3.size())
        # print(short_cut.size())
        out = x3 + short_cut
        return out


class AttentionBlock(nn.Module):
    """
    Attention gate from https://arxiv.org/abs/1804.03999.

    Assumes that the input and gate have the same shape (not including features).
    Adapted from:
        https://github.com/sfczekalski/attention_unet
    """
    def __init__(self, x_channels: int, g_channels: int, inner_channels: Optional[int] = None):
        """
        Attention gate initializer.

        Args:
            x_channels (int): Number of input features.
            g_channels (int): Number of gate features.
            inner_channels (int, optional): Number of trainable features in the attention gate.
                                            Defaults to number of input features.
        """
        super().__init__()
        if inner_channels is None: inner_channels = x_channels

        self._conv_x = nn.Sequential(
            nn.Conv2d(x_channels, inner_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(inner_channels)
        )
        self._conv_g = nn.Sequential(
            nn.Conv2d(g_channels, inner_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(inner_channels)
        )

        self._relu = nn.ReLU(inplace=True)
        self._psi = nn.Sequential(
            nn.Conv2d(inner_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        """
        Attention gate forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            g (torch.Tensor): Gate tensor.

        Returns:
            Attention gate output.
        """
        x1 = self._conv_x(x)
        g1 = self._conv_g(g)
        relu = self._relu(x1 + g1)
        psi = self._psi(relu)
        out = x * psi
        return out


class DownStack(nn.Module):
    """
    Downsampling stack for Res-UNet.

    Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
    """
    def __init__(self, in_channels: int, out_channels: int, blocks: int):
        """
        Downsampling stack initializer.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            blocks (int): Number of residual blocks.
        """
        super().__init__()
        
        modules = [ResBlock(in_channels, out_channels, ResBlockType.DOWNSAMPLER)]
        for _ in range(blocks-1):
            modules.append(ResBlock(out_channels, out_channels, ResBlockType.ENCODER))
        self._convs = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        """
        Downsampling stack forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Downsampling stack output.
        """
        out = self._convs(x)
        return out


class UpStack(nn.Module):
    """
    Upsampling stack for Res-UNet.

    Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int):
        """
        Upsampling stack initializer.

        Args:
            in_channels (int): Number of input features.
            skip_channels (int): Number of features from skip connection.
            out_channels (int): Number of output features.
            blocks (int): Number of residual blocks.
        """
        super().__init__()

        self._upsampler = ResBlock(in_channels, in_channels, ResBlockType.UPSAMPLER)
        self._attention = AttentionBlock(skip_channels, in_channels)

        modules = [ResBlock(in_channels + skip_channels, out_channels, ResBlockType.DECODER)]
        for _ in range(blocks-2):
            modules.append(ResBlock(out_channels, out_channels, ResBlockType.DECODER))
        self._convs = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        """
        Upsampling stack forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            skip (torch.Tensor): Skip connection tensor.

        Returns:
            Upsampling stack output.
        """
        x1 = self._upsampler(x)
        s1 = self._attention(skip, x1)
        x2 = torch.cat((s1, x1), dim=1)
        out = self._convs(x2)
        return out


class OutputConvBlock(nn.Module):
    """
    Output convolution block from https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2.

    Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        """
        Output convolution block initializer.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            mid_channels (int, optional): Number of intermediate features. Defaults to number of input features.
        """
        super().__init__()
        if mid_channels is None: mid_channels = in_channels

        self._convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Output convolution block forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output convolution block output.
        """
        out = self._convs(x)
        return out