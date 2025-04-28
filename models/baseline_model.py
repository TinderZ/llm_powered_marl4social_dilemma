# baseline_model.py
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Tuple, List, Optional

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict

from models.common_layers import build_conv_layers, build_fc_layers
# from models.actor_critic_lstm import ActorCriticLSTM


# Helper nn.Module layers for preprocessing (Keep these as they are)
class PermuteChannels(nn.Module):
    """Permutes tensor dimensions from (B, H, W, C) to (B, C, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)

class NormalizeUint8(nn.Module):
    """Normalizes a uint8 tensor (0-255) to a float tensor (0.0-1.0)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0


class BaselineModel(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        num_outputs: int, # LSTM hidden size
        model_config: ModelConfigDict,
        name: str,
        # --- Custom args ---
        conv_filters: List[List],
        fcnet_hiddens: List[int],
        lstm_hidden_size: int = 128,
        conv_activation: str = "relu",
        fc_activation: str = "relu",
        use_orthogonal_init: bool = True
    ):
        nn.Module.__init__(self)
        RecurrentNetwork.__init__(self, obs_space, action_space, lstm_hidden_size, model_config, name)

        self.conv_filters = conv_filters
        self.fcnet_hiddens = fcnet_hiddens
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation
        self.use_orthogonal_init = use_orthogonal_init

        # --- Encoder (保持不变) ---
        if isinstance(obs_space, gym.spaces.Dict):
             original_obs_space = obs_space.original_space
             h, w, c = original_obs_space.shape
        else:
             h, w, c = obs_space.shape

        self.preprocessor = nn.Sequential(PermuteChannels(), NormalizeUint8())
        self.conv_layers, self.conv_out_dim = build_conv_layers(input_shape=(c, h, w), conv_filters=self.conv_filters, activation=self.conv_activation)
        self.fc_layers = build_fc_layers(input_dim=self.conv_out_dim, fcnet_hiddens=self.fcnet_hiddens, activation=self.fc_activation)
        self.encoder_out_dim = self.fcnet_hiddens[-1]

        # --- LSTM Core (保持不变) ---
        self.lstm = nn.LSTMCell(self.encoder_out_dim, self.lstm_hidden_size)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.lstm.weight_ih)
            nn.init.orthogonal_(self.lstm.weight_hh)
            nn.init.constant_(self.lstm.bias_ih, 0.0)
            nn.init.constant_(self.lstm.bias_hh, 0.0)

        # --- *** Critic Head (添加回来) *** ---
        self.critic_head = nn.Linear(self.lstm_hidden_size, 1)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.constant_(self.critic_head.bias, 0.0)

        # --- *** Value Storage (添加回来) *** ---
        self._last_vf_out: Optional[torch.Tensor] = None


    def get_initial_state(self) -> List[torch.Tensor]:
        # (保持不变)
        try:
             device = next(self.parameters()).device
        except StopIteration:
             device = torch.device("cpu")
        return [
            torch.zeros(self.lstm_hidden_size, device=device),
            torch.zeros(self.lstm_hidden_size, device=device)
        ]

    @property
    def max_seq_len(self):
         # (保持不变)
        return self.model_config.get("max_seq_len", 20)


    def forward_rnn(self, inputs: torch.Tensor, state: List[torch.Tensor], seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # (大部分逻辑保持不变, 增加 value 计算和存储)
        B = state[0].shape[0]
        T = -1
        # --- Input shape handling (保持不变) ---
        if inputs.shape[0] == B: # Handle T=1 case during init
            T = 1
            inputs = inputs.unsqueeze(1) # (B, F) -> (B, 1, F)
        elif inputs.shape[0] % B == 0:
            T = inputs.shape[0] // B
            inputs = inputs.reshape(B, T, inputs.shape[-1]) # (B*T, F) -> (B, T, F)
        else:
             # Handle potential errors if shape is unexpected
             raise ValueError(f"Cannot infer time dimension T from input shape {inputs.shape} and batch size {B}")


        outputs = []
        h, c = state
        for t in range(T):
            lstm_input_t = inputs[:, t, :]
            h, c = self.lstm(lstm_input_t, (h, c))
            outputs.append(h)

        output_tensor = torch.stack(outputs, dim=1) # Shape: (B, T, hidden_size)

        # --- *** 计算并存储所有时间步的 Value *** ---
        # Apply critic head to features of all time steps
        vf_out = self.critic_head(output_tensor) # Shape: (B, T, 1)
        self._last_vf_out = vf_out.reshape(-1, 1) # Reshape to (B*T, 1) for storage/value_function

        # 返回 (B*T, hidden_size) 和最终状态 [h, c]
        return output_tensor.reshape(-1, self.lstm_hidden_size), [h, c]


    def forward(
        self,
        input_dict: ModelInputDict,
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # (保持不变)
        obs = input_dict["obs"]
        is_training = input_dict.get("is_training", True) # Get is_training if available

        if isinstance(self.obs_space, gym.spaces.Dict):
            if "obs" in obs:
                 obs = obs["obs"]

        processed_obs = self.preprocessor(obs.float())
        conv_out = self.conv_layers(processed_obs)
        encoded_features = self.fc_layers(conv_out)

        # Make sure state is correctly placed on the device of encoded_features
        # This is usually handled by RLlib, but check if issues persist
        if state and encoded_features.device != state[0].device:
             state = [s.to(encoded_features.device) for s in state]

        output_features, new_state = self.forward_rnn(encoded_features, state, seq_lens)
        return output_features, new_state

    def value_function(self) -> torch.Tensor:
        """Returns the value estimate computed in the last forward pass."""
        assert self._last_vf_out is not None, "Must call forward() first"
        # Return shape (B*T,)
        return self._last_vf_out.squeeze(-1)

# Helper nn.Module layers for preprocessing
class PermuteChannels(nn.Module):
    """Permutes tensor dimensions from (B, H, W, C) to (B, C, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)

class NormalizeUint8(nn.Module):
    """Normalizes a uint8 tensor (0-255) to a float tensor (0.0-1.0)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0