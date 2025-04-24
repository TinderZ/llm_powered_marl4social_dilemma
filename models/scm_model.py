# scm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, Tuple, List, Optional

from common_layers import build_conv_layers, build_fc_layers, get_activation_fn, normc_initializer
from moa_model import MOAModel # Inherit from MOA
from baseline_model import PermuteChannels, NormalizeUint8 # Reuse preprocessors

class SCMEncoder(nn.Module):
    """Encoder specifically for the SCM module."""
    def __init__(self, input_shape: Tuple[int, int, int], conv_filters: List[List], activation: str):
        super().__init__()
        self.preprocessor = nn.Sequential(PermuteChannels(), NormalizeUint8())
        self.conv_layers, self.output_dim = build_conv_layers(
            input_shape=input_shape,
            conv_filters=conv_filters,
            activation=activation
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        processed = self.preprocessor(obs)
        encoded = self.conv_layers(processed)
        return encoded

class ForwardModel(nn.Module):
    """Predicts the *next* SCM-encoded state."""
    def __init__(self, encoded_state_dim: int, all_actions_dim: int, lstm_hidden_dim: int, fc_hidden_dim: int = 32, activation: str = "relu"):
        super().__init__()
        act_fn = get_activation_fn(activation)
        # Inputs: Encoded state(t-1), All actions(t-1), MOA LSTM hidden(t-1), Influence reward(t-1)
        # NOTE: The original code had a slight inconsistency/complexity here regarding timing.
        # It used LSTM state from t-1, but influence reward from t-1.
        # For simplicity, we'll assume all inputs relate to time t-1 to predict state at t.
        input_dim = encoded_state_dim + all_actions_dim + lstm_hidden_dim # Removed influence input for simplicity matching paper intent more closely? Let's keep it for now to match code. + 1 # for influence reward
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.act1 = act_fn
        self.fc2 = nn.Linear(fc_hidden_dim, encoded_state_dim) # Predict next encoded state

        # Init (optional, can use default or normc like original)
        # normc_initializer(1.0)(self.fc1.weight)
        # nn.init.constant_(self.fc1.bias, 0.0)
        # normc_initializer(1.0)(self.fc2.weight) # Paper might imply relu on output? Original used relu here.
        # nn.init.constant_(self.fc2.bias, 0.0)
        self.output_activation = nn.ReLU() # Match original code's output activation

    def forward(self, encoded_now: torch.Tensor, all_actions: torch.Tensor, moa_lstm_h: torch.Tensor) -> torch.Tensor: # removed influence_reward input
        # Concatenate inputs (ensure correct shapes)
        # influence_reward = influence_reward.unsqueeze(-1) # Add feature dim if needed
        # input_vec = torch.cat([encoded_now, all_actions, moa_lstm_h, influence_reward], dim=-1)
        input_vec = torch.cat([encoded_now, all_actions, moa_lstm_h], dim=-1)
        hidden = self.act1(self.fc1(input_vec))
        predicted_next_encoded = self.output_activation(self.fc2(hidden))
        return predicted_next_encoded


class InverseModel(nn.Module):
    """Predicts the agent's *own action* based on state change."""
    # Note: The original paper's inverse model predicts the action taken between state t and t+1.
    # The original *code* predicts the *social influence reward* based on state t-1, state t, action t-1, lstm t-1.
    # We will implement the version from the *code*.
    def __init__(self, encoded_state_dim: int, all_actions_dim: int, lstm_hidden_dim: int, fc_hidden_dim: int = 32, activation: str = "relu"):
        super().__init__()
        act_fn = get_activation_fn(activation)
        # Inputs: Encoded state(t-1), Encoded state(t), All actions(t-1), MOA LSTM hidden(t-1)
        input_dim = encoded_state_dim + encoded_state_dim + all_actions_dim + lstm_hidden_dim
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.act1 = act_fn
        # Output: Predicted social influence reward (scalar)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        self.output_activation = nn.ReLU() # Match original code

    def forward(self, encoded_now: torch.Tensor, encoded_next: torch.Tensor, all_actions: torch.Tensor, moa_lstm_h: torch.Tensor) -> torch.Tensor:
        input_vec = torch.cat([encoded_now, encoded_next, all_actions, moa_lstm_h], dim=-1)
        hidden = self.act1(self.fc1(input_vec))
        predicted_influence = self.output_activation(self.fc2(hidden))
        return predicted_influence


class SocialCuriosityModule(MOAModel):
    """
    Extends MOAModel with a Social Curiosity Module (SCM) comprising
    an SCM encoder, a forward model, and an inverse model.
    Generates a social curiosity reward based on forward prediction error.
    """
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        num_other_agents: int,
        model_config: Dict, # Needs keys from MOAModel + scm_conv_filters, scm_fc_hidden_dim
    ):
        super().__init__(obs_space, action_space, num_other_agents, model_config)

        # SCM Specific Components
        h, w, c = obs_space.shape
        scm_conv_filters = model_config.get("scm_conv_filters", [[6, [3, 3], 1]]) # Example default
        scm_activation = model_config.get("scm_activation", "relu") # Example default
        fc_hidden_dim = model_config.get("scm_fc_hidden_dim", 32)

        self.scm_encoder = SCMEncoder(
            input_shape=(c, h, w),
            conv_filters=scm_conv_filters,
            activation=scm_activation
        )
        encoded_state_dim = self.scm_encoder.output_dim

        self.forward_model = ForwardModel(
            encoded_state_dim=encoded_state_dim,
            all_actions_dim=self.all_actions_dim,
            lstm_hidden_dim=self.lstm_hidden_size,
            fc_hidden_dim=fc_hidden_dim,
            activation=model_config.get("fc_activation", "relu")
        )

        self.inverse_model = InverseModel(
            encoded_state_dim=encoded_state_dim,
            all_actions_dim=self.all_actions_dim,
            lstm_hidden_dim=self.lstm_hidden_size,
            fc_hidden_dim=fc_hidden_dim,
            activation=model_config.get("fc_activation", "relu")
        )

        # --- Intermediate value storage ---
        self._scm_encoded_state: Optional[torch.Tensor] = None
        self._social_curiosity_reward: Optional[torch.Tensor] = None
        self._inverse_model_loss_input: Optional[torch.Tensor] = None # Store predicted influence

    def forward(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        state: Tuple[torch.Tensor, ...] # (h_ac, c_ac, h_moa, c_moa, prev_scm_encoded)
        ) -> Tuple[torch.Tensor, ...]: # (logits, value, moa_preds, influence_reward, curiosity_reward, inv_model_pred, new_state)
        """
        Forward pass for the SCM model.

        Args:
            obs: Observation tensor (B, H, W, C).
            prev_actions: Integer actions from t-1 (B, 1 + num_other_agents).
            state: Tuple containing LSTM states and *previous* SCM encoded state.
                   (h_ac, c_ac, h_moa, c_moa, prev_scm_encoded_state)

        Returns:
            A tuple containing:
                - Policy logits (B, num_actions).
                - Value estimate (B, 1).
                - MOA predicted logits for other agents (B, num_other_agents * num_actions).
                - Social influence reward (placeholder, B, ).
                - Social curiosity reward (forward model error, B, ).
                - Inverse model prediction (predicted influence, B, 1).
                - New combined state tuple (h_ac, c_ac, h_moa, c_moa, current_scm_encoded).
        """
        # Unpack state
        h_ac_old, c_ac_old, h_moa_old, c_moa_old, prev_scm_encoded_state = state

        # === Run MOA forward pass first ===
        moa_state = (h_ac_old, c_ac_old, h_moa_old, c_moa_old)
        policy_logits, value, moa_predicted_logits, _, (h_ac_new, c_ac_new, h_moa_new, c_moa_new) = \
            super().forward(obs, prev_actions, moa_state)
        # Note: MOAModel.forward currently returns placeholder influence reward.

        # === SCM Calculations ===
        # 1. Encode current observation using SCM encoder
        current_scm_encoded_state = self.scm_encoder(obs)
        self._scm_encoded_state = current_scm_encoded_state # Store for potential access

        # 2. Prepare inputs for Forward/Inverse models (using state from t-1)
        # Flatten previous actions to one-hot
        prev_actions_one_hot = F.one_hot(prev_actions, num_classes=self.num_actions).float()
        flat_prev_actions_one_hot = prev_actions_one_hot.view(prev_actions.shape[0], -1)

        # Use MOA hidden state from t-1 (h_moa_old) - stop gradients? Original did.
        moa_hidden_prev = h_moa_old.detach() # Detach to prevent SCM loss affecting MOA LSTM much

        # Influence reward from t-1 (needed by original inverse model code)
        # --> This isn't available easily. The inverse model *predicts* it.
        # --> Let's stick to paper/code: inverse predicts influence.
        # --> Forward model: Use prev_scm_encoded, prev_actions, prev_moa_hidden
        predicted_next_scm_encoded = self.forward_model(
            prev_scm_encoded_state, flat_prev_actions_one_hot, moa_hidden_prev
        )

        # Inverse model: Use prev_scm_encoded, current_scm_encoded, prev_actions, prev_moa_hidden
        # Predicts influence reward at step t-1 based on transition t-1 -> t
        predicted_influence_reward = self.inverse_model(
             prev_scm_encoded_state, current_scm_encoded_state.detach(), # Detach current? Avoid cycle?
             flat_prev_actions_one_hot, moa_hidden_prev
        )
        self._inverse_model_loss_input = predicted_influence_reward # Store for loss calculation

        # 3. Compute Social Curiosity Reward (Forward Model Error)
        # MSE between predicted(t) and actual(t) SCM encoded states
        forward_loss = F.mse_loss(predicted_next_scm_encoded, current_scm_encoded_state.detach(), reduction='none')
        # Reduce mean over the feature dimension
        curiosity_reward_raw = forward_loss.mean(dim=-1) # Shape (B,)
        self._social_curiosity_reward = curiosity_reward_raw * 0.5 # Match original scaling? Check SCMLoss


        # === Combine Outputs ===
        new_state = (h_ac_new, c_ac_new, h_moa_new, c_moa_new, current_scm_encoded_state)
        placeholder_influence_reward = torch.zeros_like(self._social_curiosity_reward)

        return (
            policy_logits,
            value,
            moa_predicted_logits,
            placeholder_influence_reward, # Placeholder for influence reward
            self._social_curiosity_reward,
            self._inverse_model_loss_input, # Predicted influence for loss calc
            new_state
        )


    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Returns the initial LSTM states and a zero tensor for the initial SCM encoded state."""
        h_ac, c_ac, h_moa, c_moa = super().get_initial_state(batch_size)
        # Need initial state for the SCM encoder output placeholder
        # Determine device from one of the LSTM states
        device = h_ac.device
        scm_initial_encoded = torch.zeros(batch_size, self.scm_encoder.output_dim, device=device)
        return h_ac, c_ac, h_moa, c_moa, scm_initial_encoded

    # --- Accessor methods for internal SCM values (if needed by algorithm) ---
    def social_curiosity_reward(self) -> Optional[torch.Tensor]:
        """Returns the raw curiosity reward (forward prediction error) from the last forward pass."""
        return self._social_curiosity_reward

    def inverse_model_prediction(self) -> Optional[torch.Tensor]:
        """Returns the output of the inverse model (predicted influence) from the last forward pass."""
        return self._inverse_model_loss_input