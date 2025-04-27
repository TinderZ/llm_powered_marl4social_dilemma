# moa_lstm.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class MoaLSTM(nn.Module):
    """
    LSTM for the Model of Other Agents (MOA) head. Predicts actions of other agents.
    """
    def __init__(
        self,
        input_size: int,          # Size of the encoded observation from MOA's FC layers
        all_actions_size: int,    # Size of concatenated one-hot actions (own + others)
        hidden_size: int,
        num_other_agents: int,
        other_agent_action_dim: int,
        use_orthogonal_init: bool = True
    ):
        """
        Args:
            input_size: Dimension of the input features (from MOA FC layers).
            all_actions_size: Dimension of the concatenated one-hot actions input.
            hidden_size: Size of the LSTM hidden state and cell state.
            num_other_agents: Number of other agents being modeled.
            other_agent_action_dim: Action dimension for a single other agent.
            use_orthogonal_init: Whether to use orthogonal initialization.
        """
        super().__init__()
        self.input_size = input_size
        self.all_actions_size = all_actions_size
        self.hidden_size = hidden_size
        self.num_other_agents = num_other_agents
        self.other_agent_action_dim = other_agent_action_dim
        self.output_size = num_other_agents * other_agent_action_dim

        # LSTM input is concatenation of encoded obs and all actions
        self.lstm = nn.LSTMCell(input_size + all_actions_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, self.output_size)

        # Initialization
        if use_orthogonal_init:
            nn.init.orthogonal_(self.lstm.weight_ih)
            nn.init.orthogonal_(self.lstm.weight_hh)
            nn.init.constant_(self.lstm.bias_ih, 0.0)
            nn.init.constant_(self.lstm.bias_hh, 0.0)
            nn.init.orthogonal_(self.output_head.weight, gain=1.0) # Standard gain for linear
            nn.init.constant_(self.output_head.bias, 0.0)
        # else: add alternative initialization if needed

    def forward(self, obs_enc: torch.Tensor, all_actions_one_hot: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the MOA LSTM.

        Args:
            obs_enc: Encoded observation from MOA FC layers, shape (batch_size, input_size).
            all_actions_one_hot: Concatenated one-hot encoded actions (own + others)
                                 from the *previous* timestep, shape (batch_size, all_actions_size).
            state: Tuple containing the hidden state (h) and cell state (c),
                   each of shape (batch_size, hidden_size).

        Returns:
            A tuple containing:
                - Predicted action logits for other agents, shape (batch_size, num_other_agents * other_agent_action_dim).
                - New LSTM state (h_new, c_new).
        """
        h_old, c_old = state
        lstm_input = torch.cat([obs_enc, all_actions_one_hot], dim=-1)
        h_new, c_new = self.lstm(lstm_input, (h_old, c_old))

        # Predict logits for other agents' actions at the *next* timestep
        other_agent_logits = self.output_head(h_new)

        return other_agent_logits, (h_new, c_new)

    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns zeroed initial states for the LSTM."""
        # device = self.lstm.weight_ih.device
        # h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        # c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros(batch_size, self.hidden_size)
        return h0, c0