"""
Heatmap generators.

Adapted from:
        https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
        https://github.com/nttcom/WASB-SBDT
"""
from typing import Tuple
import numpy as np


def generate_heatmap(
        shape: Tuple[int, int],
        center: Tuple[int, int],
        radius: int = 1,
        type: np.dtype = np.float32):
    """
    Generate binary heatmap.
    
    Args:
        shape (Tuple[int, int]): output heatmap shape as height, width
        center (Tuple[int, int]): ball center in image coordinates as x, y (zero-indexed)
        radius (int): radius of generated circle
        type (np.dtype): heatmap datatype (defaults to np.float32)

    Returns:
        Binary heatmap for ball.
    """
    h, w = shape
    bx, by = center
    if bx < 0 or by < 0:
        return np.zeros((h, w), dtype=type)
    
    x, y = np.meshgrid(np.linspace(1, w, num=w), np.linspace(1, h, num=h))
    distmap = (x - (bx + 1))**2 + (y - (by + 1))**2
    heatmap = np.zeros_like(distmap)
    heatmap[distmap <= radius**2] = 1
    return heatmap.astype(type)


if __name__ == '__main__':
    print(generate_heatmap(shape=(5, 7), center=(3, 2)))