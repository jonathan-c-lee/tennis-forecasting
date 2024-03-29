"""Focal loss described in https://arxiv.org/abs/1708.02002."""
import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss from https://arxiv.org/abs/1708.02002.
    
    Adapted From:
        https://github.com/nttcom/WASB-SBDT
    """
    def __init__(self, gamma: float = 2.0):
        """
        Binary focal loss initializer.

        Args:
            gamma (float): Gamma coefficient for focal loss.
        """
        super().__init__()
        self._gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Binary focal loss forward pass.

        Args:
            inputs (torch.Tensor): Predictions tensor.
            targets (torch.Tensor): Labels tensor.

        Returns:
            Binary focal loss.
        """
        loss = (targets * (1 - inputs)**self._gamma * torch.log(inputs) +
                (1 - targets) * inputs**self._gamma * torch.log(1 - inputs))
        loss = torch.mean(-loss)
        return loss