"""
Ball trajectory prediction model.

Architecture from https://www.mdpi.com/2313-433X/9/5/99.
"""
import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    """Ball trajectory prediction model from https://www.mdpi.com/2313-433X/9/5/99."""
    def __init__(
            self,
            output_size: int,
            position_dim: int,
            pose_size: int,
            pose_dim: int,
            hidden_dim: int,
            lstm_layers: int,
            dropout: float):
        """
        TrajectoryPredictor initializer.

        Args:
            output_size (int): Number of output frames.
            position_dim (int): Number of dimensions that players' positions are mapped to.
            pose_size (int): Dimension of players' poses.
            pose_dim (int): Number of dimensions that players' poses are mapped to.
            hidden_dim (int): Number of features in LSTM hidden state.
            lstm_layers (int): Number of layers in LSTM.
            dropout (float): LSTM dropout value.
        """
        super().__init__()
        self._hidden_dim = hidden_dim

        self._player_fc = nn.Linear(4, position_dim)
        self._pose_fc = nn.Linear(2*pose_size, pose_dim)
        self._lstm = nn.LSTM(
            input_size=2+position_dim+pose_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        self._output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*output_size)
        )

    def forward(
            self,
            ball_positions: torch.Tensor,
            players_positions: torch.Tensor,
            players_poses: torch.Tensor):
        """
        Ball trajectory prediction model forward pass.

        Args:
            ball_positions (torch.Tensor): Sequence of ball positions.
            players_positions (torch.Tensor): Sequence of players' positions.
            players_poses (torch.Tensor): Sequence of players' poses.
        
        Returns:
            Future ball positions.
        """
        players_positions = self._player_fc(players_positions)
        players_poses = self._pose_fc(players_poses)
        input = torch.cat((ball_positions, players_positions, players_poses), dim=-1)

        lstm_out, _ = self._lstm(input)
        out = self._output_fc(lstm_out[:, -1, :])
        return out


class TrajectoryBaseline(nn.Module):
    """Baseline ball trajectory prediction model using only ball position."""
    def __init__(
            self,
            output_size: int,
            hidden_dim: int,
            lstm_layers: int,
            dropout: float):
        """
        TrajectoryBaseline initializer.

        Args:
            output_size (int): Number of output frames.
            hidden_dim (int): Number of features in LSTM hidden state.
            lstm_layers (int): Number of layers in LSTM.
            dropout (float): LSTM dropout value.
        """
        super().__init__()
        self._hidden_dim = hidden_dim

        self._lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        self._fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*output_size)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Ball trajectory prediction model forward pass.

        Args:
            x (torch.Tensor): Sequence of ball positions.

        Returns:
            Future ball positions.
        """
        lstm_out, _ = self._lstm(x)
        out = self._fc(lstm_out[:, -1, :])
        return out

