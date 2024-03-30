"""Detectors."""
import numpy as np
import cv2
import torch


def detect_ball(heatmap: torch.Tensor, threshold: float = 0.5):
        """
        Detect ball in heatmap.

        Args:
            heatmap (torch.Tensor): Raw heatmap.
            threshold (float): Threshold for binary map conversion.
        
        Returns:
            Ball center coordinates.
        """
        heatmap = np.squeeze(heatmap.detach().numpy())
        if np.max(heatmap) < threshold: return (-1, -1)

        _, binary_map = cv2.threshold(heatmap, threshold, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        center = (int(x + w/2), int(y + h/2))
        return center