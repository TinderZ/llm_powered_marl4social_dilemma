# baseline_model.py
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Tuple, List, Optional

from common_layers import build_conv_layers, build_fc_layers
from actor_critic_lstm import ActorCriticLSTM

class BaselineModel(nn.Module):
    """
    Baseline Actor-Critic model with CNN encoder and LSTM core.
    Mimics the structure of the original baseline_model.py.
    """
    def __init__(
        self,
        obs_space: gym.spaces.Box, # Expects Box(H, W, C)
        action_space: gym.spaces.Discrete,
        conv_filters: List[List], # [[out_channels, kernel, stride], ...]
        fcnet_hiddens: List[int],
        lstm_hidden_size: int = 128,
        conv_activation: str = "relu",
        fc_activation: str = "relu",
        use_orthogonal_init: bool = True
    ):
        """
        Args:
            obs_space: Observation space (Box).
            action_space: Action space (Discrete).
            conv_filters: Configuration for convolutional layers.
            fcnet_hiddens: Configuration for fully connected layers after CNN.
            lstm_hidden_size: Size of the LSTM hidden state.
            conv_activation: Activation for conv layers.
            fc_activation: Activation for FC layers.
            use_orthogonal_init: Whether to use orthogonal initialization.
        """
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actions = action_space.n
        self.lstm_hidden_size = lstm_hidden_size

        # --- Encoder ---
        # Assuming obs shape is (H, W, C) - need to permute to (C, H, W) for PyTorch Conv2d
        h, w, c = obs_space.shape
        self.preprocessor = nn.Sequential(
            # Permute and normalize image observations
            PermuteChannels(), # Custom layer to change HWC to CHW
            NormalizeUint8() # Custom layer to scale 0-255 to 0-1
        )
        # Build CNN layers
        self.conv_layers, self.conv_out_dim = build_conv_layers(
            input_shape=(c, h, w), # CHW
            conv_filters=conv_filters,
            activation=conv_activation,
            # init_std = 1.0 # Or use default Kaiming
        )
        # Build FC layers after CNN
        self.fc_layers = build_fc_layers(
            input_dim=self.conv_out_dim,
            fcnet_hiddens=fcnet_hiddens,
            activation=fc_activation,
            # init_std = 1.0
        )
        self.encoder_out_dim = fcnet_hiddens[-1]

        # --- Actor-Critic LSTM Core ---
        self.lstm_core = ActorCriticLSTM(
            input_size=self.encoder_out_dim,
            hidden_size=lstm_hidden_size,
            num_actions=self.num_actions,
            use_orthogonal_init=use_orthogonal_init
        )

        # --- Value storage for external access ---
        self._last_value: Optional[torch.Tensor] = None

    def forward(self, obs: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: Encode observation, pass through LSTM, get action logits and value.

        Args:
            obs: Observation tensor, shape (batch_size, H, W, C).
                 Needs pre-processing (permute, normalize).
            state: LSTM state tuple (h, c).

        Returns:
            A tuple containing:
                - Action logits (batch_size, num_actions).
                - Value estimate (batch_size, 1).
                - New LSTM state (h_new, c_new).
        """
        # 1. Preprocess and Encode Observation
        processed_obs = self.preprocessor(obs) # B, C, H, W
        conv_out = self.conv_layers(processed_obs) # B, conv_out_dim
        encoded_features = self.fc_layers(conv_out) # B, encoder_out_dim

        # 2. Pass through LSTM Core
        logits, value, new_state = self.lstm_core(encoded_features, state)

        # Store value for potential external access (e.g., by algorithm)
        self._last_value = value

        return logits, value, new_state # Return logits, value, and new state

    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the initial LSTM state."""
        return self.lstm_core.get_initial_state(batch_size)

    def value_function(self) -> Optional[torch.Tensor]:
        """Returns the value estimate from the most recent forward pass."""
        # .detach() prevents gradients from flowing back further if used externally
        return self._last_value.detach() if self._last_value is not None else None


# Helper nn.Module layers for preprocessing
class PermuteChannels(nn.Module):
    """Permutes tensor dimensions from (B, H, W, C) to (B, C, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)

class NormalizeUint8(nn.Module):
    """Normalizes a uint8 tensor (0-255) to a float tensor (0.0-1.0)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0