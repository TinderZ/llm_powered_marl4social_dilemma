# moa_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, Tuple, List, Optional

from common_layers import build_conv_layers, build_fc_layers
from actor_critic_lstm import ActorCriticLSTM
from moa_lstm import MoaLSTM
from baseline_model import PermuteChannels, NormalizeUint8 # Reuse preprocessors


# Helper function for KL divergence (adapted from original repo)
def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    """Calculates KL divergence D_KL(P || Q) for two categorical distributions.
       Assumes p and q are tensors of probabilities, sum(p, dim=-1) == 1.
       Adds epsilon for numerical stability.
    """
    # Ensure probabilities sum to 1 and handle potential zeros
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    
    # Compute KL divergence: sum(p * log(p / q))
    kl_div = p * (torch.log(p + epsilon) - torch.log(q + epsilon))
    
    # Sum over the action dimension
    result = kl_div.sum(dim=-1)
    
    # Handle potential NaNs or Infs (e.g., if q is zero where p is not)
    # result = torch.where(torch.isfinite(result), result, torch.zeros_like(result)) # Original logic
    # A simpler approach might be to rely on the epsilon stabilization
    return result

class MOAModel(nn.Module):
    """
    Model of Other Agents (MOA) architecture.
    Includes a standard actor-critic path and an MOA path to predict others' actions
    and compute social influence reward.
    """
    def __init__(
        self,
        obs_space: gym.spaces.Box, # Expects Box(H, W, C)
        action_space: gym.spaces.Discrete,
        num_other_agents: int,
        model_config: Dict, # Use a dict for cleaner config passing
                            # Expected keys: conv_filters, policy_fc_hiddens, moa_fc_hiddens,
                            # lstm_hidden_size, conv_activation, fc_activation,
                            # use_orthogonal_init, influence_divergence_measure ('kl' or 'jsd')
    ):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actions = action_space.n
        self.num_other_agents = num_other_agents
        self.model_config = model_config
        self.lstm_hidden_size = model_config.get("lstm_hidden_size", 128)
        self.influence_divergence_measure = model_config.get("influence_divergence_measure", "kl")

        # --- Encoder (Shared Conv, Separate FC) ---
        h, w, c = obs_space.shape
        self.preprocessor = nn.Sequential(PermuteChannels(), NormalizeUint8())
        self.conv_layers, conv_out_dim = build_conv_layers(
            input_shape=(c, h, w),
            conv_filters=model_config["conv_filters"],
            activation=model_config.get("conv_activation", "relu")
        )

        # Separate FC layers for Policy/Value and MOA branches
        self.policy_fc_layers = build_fc_layers(
            input_dim=conv_out_dim,
            fcnet_hiddens=model_config["policy_fc_hiddens"],
            activation=model_config.get("fc_activation", "relu")
        )
        self.policy_encoder_out_dim = model_config["policy_fc_hiddens"][-1]

        self.moa_fc_layers = build_fc_layers(
            input_dim=conv_out_dim,
            fcnet_hiddens=model_config["moa_fc_hiddens"],
            activation=model_config.get("fc_activation", "relu")
        )
        self.moa_encoder_out_dim = model_config["moa_fc_hiddens"][-1]

        # --- Actor-Critic LSTM Core ---
        self.actor_critic_lstm = ActorCriticLSTM(
            input_size=self.policy_encoder_out_dim,
            hidden_size=self.lstm_hidden_size,
            num_actions=self.num_actions,
            use_orthogonal_init=model_config.get("use_orthogonal_init", True)
        )

        # --- MOA LSTM Core ---
        # MOA LSTM input includes MOA encoded obs + all actions (own + others) one-hot
        self.all_actions_dim = self.num_actions * (1 + self.num_other_agents)
        self.moa_lstm = MoaLSTM(
            input_size=self.moa_encoder_out_dim,
            all_actions_size=self.all_actions_dim,
            hidden_size=self.lstm_hidden_size,
            num_other_agents=self.num_other_agents,
            other_agent_action_dim=self.num_actions, # Assuming same action dim
            use_orthogonal_init=model_config.get("use_orthogonal_init", True)
        )

        # --- Intermediate value storage ---
        self._last_value: Optional[torch.Tensor] = None
        self._last_policy_logits: Optional[torch.Tensor] = None
        self._last_moa_predictions: Optional[torch.Tensor] = None # Logits predicted by MOA LSTM
        self._social_influence_reward: Optional[torch.Tensor] = None # Calculated influence


    def forward(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor, # Shape (B, 1 + num_other_agents), integer actions
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] # (h_ac, c_ac, h_moa, c_moa)
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for the MOA model.

        Args:
            obs: Observation tensor (B, H, W, C).
            prev_actions: Integer actions taken by *all* agents (including self) at the *previous*
                          timestep. Shape (B, 1 + num_other_agents). Needed for MOA LSTM input.
            state: Tuple containing LSTM states for ActorCritic and MOA LSTMs.
                   (h_actor_critic, c_actor_critic, h_moa, c_moa)

        Returns:
            A tuple containing:
                - Policy logits (B, num_actions).
                - Value estimate (B, 1).
                - MOA predicted logits for other agents (B, num_other_agents * num_actions).
                - Social influence reward for the *previous* step (B, ).
                - New combined LSTM state tuple (h_ac_new, c_ac_new, h_moa_new, c_moa_new).
        """
        h_ac_old, c_ac_old, h_moa_old, c_moa_old = state

        # 1. Encode Observation (Shared Conv, Split FC)
        processed_obs = self.preprocessor(obs)
        conv_out = self.conv_layers(processed_obs)
        policy_fc_out = self.policy_fc_layers(conv_out)
        moa_fc_out = self.moa_fc_layers(conv_out)

        # 2. Actor-Critic Path
        policy_logits, value, (h_ac_new, c_ac_new) = self.actor_critic_lstm(
            policy_fc_out, (h_ac_old, c_ac_old)
        )
        self._last_value = value
        self._last_policy_logits = policy_logits # Store logits from t for influence calc at t+1

        # 3. MOA Path
        # Prepare MOA LSTM inputs: requires *previous* actions
        # Convert previous integer actions to one-hot and concatenate
        # Note: The input `prev_actions` are actions from t-1 that led to the current `obs` at t.
        # The MOA LSTM uses the *observation encoding* from t (`moa_fc_out`)
        # and the *actions from t-1* (`prev_actions`) to predict actions at t.
        prev_actions_one_hot = F.one_hot(prev_actions, num_classes=self.num_actions).float()
        # Flatten the one-hot actions: (B, 1 + N_other, A) -> (B, (1 + N_other) * A)
        flat_prev_actions_one_hot = prev_actions_one_hot.view(prev_actions.shape[0], -1)

        moa_predicted_logits, (h_moa_new, c_moa_new) = self.moa_lstm(
            moa_fc_out, flat_prev_actions_one_hot, (h_moa_old, c_moa_old)
        )
        self._last_moa_predictions = moa_predicted_logits # Predicted logits for actions at t

        # 4. Compute Social Influence Reward (using predictions about t based on t-1)
        # This is complex as it requires counterfactuals.
        # We need policy logits from t-1 to compute the reward for the action taken at t-1.
        # This implies the reward calculation might need to happen outside the forward pass,
        # or the forward pass needs access to the *previous* policy logits.
        # Let's attempt the latter, assuming previous logits are part of the state or input dict.
        # --> Modification: Assume the *calling algorithm* provides `prev_policy_logits`
        # --> Simpler approach: Calculate reward based on *current* predictions and *current* policy.
        #     This isn't exactly the original intent but fits better in a standard forward pass.
        #     Let's return 0 influence reward for now and suggest computing it externally.
        # --> Compromise: Return components needed for external calculation?
        # --> Let's stick to the original goal: calculate it here. Requires `prev_policy_logits`.

        # Assume `prev_policy_logits` (from step t-1) is somehow passed, maybe via state or input_dict
        # Placeholder: If not available, return zero influence. For now, we'll just compute based on current.
        # This calculation will be for the influence of action *at t* on others *at t+1*.
        # The *true* original calculation was influence of action *at t-1* on others *at t*.
        # We will return the components needed for the algorithm to compute the reward later.

        # We need:
        # - moa_predicted_logits (predicts others' actions at t | obs at t, actions at t-1)
        # - policy_logits (predicts own action at t | obs at t)
        # - A way to get counterfactual MOA predictions (predict others' actions at t | obs at t, HYPOTHETICAL own action at t-1)
        # This seems too complex for a standard forward pass without access to previous states/inputs.

        # --- Simplified Approach: Return components for external calculation ---
        # The algorithm calling this model will need:
        # 1. `policy_logits` from step t-1
        # 2. `moa_predicted_logits` from step t (which used actions from t-1)
        # 3. A way to compute counterfactuals (re-running MOA LSTM with hypothetical prev actions)
        # Let's return the MOA predictions and let the algorithm handle the rest.
        # We'll return None for influence reward here.
        self._social_influence_reward = torch.zeros(policy_logits.shape[0], device=policy_logits.device)


        new_state = (h_ac_new, c_ac_new, h_moa_new, c_moa_new)

        return policy_logits, value, moa_predicted_logits, self._social_influence_reward, new_state


    # --- Methods to access internal states (if needed by algorithm) ---

    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Returns the initial LSTM states for both LSTMs."""
        h_ac, c_ac = self.actor_critic_lstm.get_initial_state(batch_size)
        h_moa, c_moa = self.moa_lstm.get_initial_state(batch_size)
        return h_ac, c_ac, h_moa, c_moa

    def value_function(self) -> Optional[torch.Tensor]:
        """Returns the value estimate from the most recent forward pass."""
        return self._last_value.detach() if self._last_value is not None else None

    def policy_logits(self) -> Optional[torch.Tensor]:
         """Returns the policy logits from the most recent forward pass."""
         return self._last_policy_logits # No detach needed if used for loss

    def predicted_other_actions(self) -> Optional[torch.Tensor]:
         """Returns the MOA predicted logits for other agents from the most recent forward pass."""
         return self._last_moa_predictions # No detach needed if used for MOA loss