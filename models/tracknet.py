"""
TrackNet Attention Res-UNet model for ball tracking.

The model combines attention, ResNet, and UNet.
Base TrackNet architecture from https://ieeexplore.ieee.org/document/9302757.
Attention architecture from https://arxiv.org/abs/1804.03999.
Res-UNet architecture from https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2.
"""
from typing import List
import numpy as np
import cv2
import torch
import torch.nn as nn

from models.tracknet_modules import *


class TrackNet(nn.Module):
    """
    TrackNet model for ball tracking.

    Uses an Attention Res-UNet architecture.
    Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
        https://github.com/sfczekalski/attention_unet
    """
    _score_threshold = 0.5

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            neck_channels: int = 64,
            blocks: List[int] = [3, 3, 4, 3],
            channels: List[int] = [16, 32, 64, 128]):
        """
        TrackNet initializer.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            neck_channels (int): Number of output features from input convolution block.
            blocks (List[int]): Number of blocks in each residual stack.
            channels (List[int]): Number of channels for each residual stack.
        """
        super().__init__()

        if len(blocks) != 4 or len(channels) != 4:
            raise ValueError('There must be 4 blocks and 4 channels')

        self._conv_in = InputConvBlock(in_channels, neck_channels)
        self._down1 = DownStack(neck_channels, channels[0], blocks[0])
        self._down2 = DownStack(channels[0], channels[1], blocks[1])
        self._down3 = DownStack(channels[1], channels[2], blocks[2])
        self._down4 = DownStack(channels[2], channels[3], blocks[3])

        self._up1 = UpStack(channels[3], channels[2], channels[2], blocks[2])
        self._up2 = UpStack(channels[2], channels[1], channels[1], blocks[1])
        self._up3 = UpStack(channels[1], channels[0], channels[0], blocks[0])
        self._up4 = ResBlock(channels[0], channels[0], ResBlockType.UPSAMPLER)
        self._conv_out = OutputConvBlock(channels[0], out_channels, mid_channels=channels[1])

    def forward(self, x: torch.Tensor):
        """
        TrackNet forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            TrackNet output.
        """
        x1 = self._conv_in(x)
        x2 = self._down1(x1)
        x3 = self._down2(x2)
        x4 = self._down3(x3)
        x5 = self._down4(x4)

        x6 = self._up1(x5, x4)
        x7 = self._up2(x6, x3)
        x8 = self._up3(x7, x2)
        x9 = self._up4(x8)
        out = self._conv_out(x9)
        return out
    
    @classmethod
    def detect_ball(cls, heatmap: torch.Tensor):
        """
        Detect ball in heatmap.

        Args:
            heatmap (torch.Tensor): Raw heatmap.
        
        Returns:
            Ball center coordinates.
        """
        heatmap = np.squeeze(heatmap.numpy())
        if np.max(heatmap) < cls._score_threshold: return (-1, -1)

        _, binary_map = cv2.threshold(heatmap, cls._score_threshold, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        moment = cv2.moments(contour)
        center = (int(moment['m10'] // moment['m00']), int(moment['m01'] // moment['m00']))
        return center