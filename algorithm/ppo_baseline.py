import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Union, Optional

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for processing agent observations."""
    def __init__(self, observation_shape):
        super(CNNFeatureExtractor, self).__init__()
        # Input shape: [B, H, W, C] -> Convert to [B, C, H, W] for PyTorch
        self.conv1 = nn.Conv2d(observation_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # Calculate output size after convolutions for the FC layer
        h_out = observation_shape[0] // 4  # After 2 stride-2 convs
        w_out = observation_shape[1] // 4
        self.fc_input_size = 64 * h_out * w_out

        with torch.no_grad():
            dummy_input = torch.zeros(1, observation_shape[2], observation_shape[0], observation_shape[1])
            conv_output = self.conv3(self.conv2(self.conv1(dummy_input)))
            self.fc_input_size = int(np.prod(conv_output.size()[1:]))  # Flattened size

        #print(f'Dynamically calculated fc_input_size: {self.fc_input_size}')

        #print(f'h_out: {h_out}, w_out: {w_out}, fc_input_size: {self.fc_input_size}')
        # input()
        
        self.fc = nn.Linear(self.fc_input_size, 512)
        
    def forward(self, x):
        # Convert [B, H, W, C] to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).float() / 255.0  # Normalize pixel values
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)   # -1, self.fc_input_size
        # print(f'{x.size(0)}, {x.size(1)}')
        x = F.relu(self.fc(x))
        
        return x

class PPOPolicy(nn.Module):
    """PPO Policy Network with Actor and Critic heads."""
    def __init__(self, observation_shape, num_actions):
        super(PPOPolicy, self).__init__()
        
        # Feature extractor shared between actor and critic
        self.feature_extractor = CNNFeatureExtractor(observation_shape)
        
        # Actor (policy) head
        self.actor = nn.Linear(512, num_actions)
        
        # Critic (value) head
        self.critic = nn.Linear(512, 1)
        
    def forward(self, obs):
        features = self.feature_extractor(obs)
        
        # Actor: Policy distribution
        logits = self.actor(features)
        
        # Critic: Value function
        value = self.critic(features)
        
        return logits, value
    
    def get_value(self, obs):
        features = self.feature_extractor(obs)
        return self.critic(features)
    
    def get_action_and_value(self, obs, action=None):
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        value = self.critic(features)
        
        return action, log_prob, entropy, value

class PPOAgent:
    """PPO Agent implementation."""
    def __init__(
        self,
        observation_shape,
        num_actions,
        lr=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_epochs=10
    ):
        self.device = device
        self.policy = PPOPolicy(observation_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).to(self.device)
            logits, _ = self.policy(obs_tensor)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
            return action.cpu().item()
    
    def update(self, rollouts):
        """Update policy using the collected rollouts."""
        obs_batch = torch.FloatTensor(rollouts["obs"]).to(self.device)
        action_batch = torch.LongTensor(rollouts["actions"]).to(self.device)
        old_log_probs_batch = torch.FloatTensor(rollouts["log_probs"]).to(self.device)
        returns_batch = torch.FloatTensor(rollouts["returns"]).to(self.device)
        advantage_batch = torch.FloatTensor(rollouts["advantages"]).to(self.device)
        
        # Normalize advantages
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            # Get new action distributions and values
            _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(obs_batch, action_batch)
            
            # Calculate PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns_batch)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = values[t+1]
                
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + values
        return advantages, returns
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

class PPOBaseline:
    """
    PPO Baseline implementation for multi-agent environments.
    This class coordinates multiple PPO agents, one per agent in the environment.
    """
    def __init__(
        self,
        observation_shape,
        num_actions,
        num_agents,
        lr=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        shared_policy=True  # Whether to use a shared policy for all agents
    ):
        self.num_agents = num_agents
        self.shared_policy = shared_policy
        self.device = device
        
        if shared_policy:
            # Create a single agent that will be used for all agents
            self.agents = {
                "shared": PPOAgent(
                    observation_shape,
                    num_actions,
                    lr=lr,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_ratio=clip_ratio,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    max_grad_norm=max_grad_norm,
                    device=device
                )
            }
        else:
            # Create a separate agent for each environment agent
            self.agents = {}
            for i in range(num_agents):
                agent_id = f"agent_{i}"
                self.agents[agent_id] = PPOAgent(
                    observation_shape,
                    num_actions,
                    lr=lr,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_ratio=clip_ratio,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    max_grad_norm=max_grad_norm,
                    device=device
                )
    
    def get_actions(self, observations, deterministic=False):
        """Get actions for all agents based on their observations."""
        actions = {}
        
        if self.shared_policy:
            # Use the shared policy for all agents
            for agent_id, obs in observations.items():
                actions[agent_id] = self.agents["shared"].get_action(obs, deterministic)
        else:
            # Use each agent's policy
            for agent_id, obs in observations.items():
                if agent_id in self.agents:
                    actions[agent_id] = self.agents[agent_id].get_action(obs, deterministic)
        
        return actions
    
    def update(self, rollouts_by_agent):
        """Update policies for all agents using their collected rollouts."""
        metrics = {}
        
        if self.shared_policy:
            # Update the shared policy with combined rollouts from all agents
            combined_rollouts = self._combine_agent_rollouts(rollouts_by_agent)
            metrics["shared"] = self.agents["shared"].update(combined_rollouts)
        else:
            # Update each agent's policy separately
            for agent_id, agent in self.agents.items():
                if agent_id in rollouts_by_agent:
                    metrics[agent_id] = agent.update(rollouts_by_agent[agent_id])
        print(metrics)
        
        return metrics
    
    def _combine_agent_rollouts(self, rollouts_by_agent):
        """Combine rollouts from all agents for updating a shared policy."""
        combined = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "returns": [],
            "advantages": [],
        }
        
        for agent_id, rollout in rollouts_by_agent.items():
            for key in combined:
                combined[key].extend(rollout[key])
                
        # Convert lists to numpy arrays
        for key in combined:
            combined[key] = np.array(combined[key])
            
        return combined
    
    def save(self, path_prefix):
        """Save model checkpoints for all agents."""
        if self.shared_policy:
            self.agents["shared"].save(f"{path_prefix}_shared.pt")
        else:
            for agent_id, agent in self.agents.items():
                agent.save(f"{path_prefix}_{agent_id}.pt")
    
    def load(self, path_prefix):
        """Load model checkpoints for all agents."""
        if self.shared_policy:
            self.agents["shared"].load(f"{path_prefix}_shared.pt")
        else:
            for agent_id, agent in self.agents.items():
                agent.load(f"{path_prefix}_{agent_id}.pt")