import os
import time
import argparse
import numpy as np
import torch
import multiprocessing as mp
from collections import defaultdict
import gymnasium as gym
from envs.cleanup_env import CleanupEnv as make_cleanup_env
from algorithm.ppo_baseline import PPOBaseline

class RolloutCollector:
    """Collects rollouts from environment using current policy."""
    def __init__(
        self, 
        env_fn, 
        policy,
        num_envs,
        rollout_length,
        gamma=0.99,
        gae_lambda=0.95
    ):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.policy = policy
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = policy.device
        
        # Reset all environments
        self.observations = {}
        for env_idx in range(num_envs):
            obs, _ = self.envs[env_idx].reset()
            if env_idx == 0:  # Initialize observation dict based on first env
                for agent_id in obs.keys():
                    self.observations[agent_id] = []
            
            for agent_id in obs.keys():
                self.observations[agent_id].append(obs[agent_id])
    
    def default_rollout_dict(self):
        """Default structure for storing rollouts."""
        return {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "advantages": [],
            "returns": [],
            "log_probs": [],
        }
    
    def collect(self):
        """Collect rollouts from all environments."""
        rollouts_by_agent = defaultdict(self.default_rollout_dict)
        
        for _ in range(self.rollout_length):
            # Get actions from policy
            actions_by_env = []
            
            for env_idx in range(len(self.envs)):
                obs_dict = {agent_id: self.observations[agent_id][env_idx] for agent_id in self.observations}
                actions = self.policy.get_actions(obs_dict)
                actions_by_env.append(actions)
            
            # Step environments and collect data
            for env_idx, env in enumerate(self.envs):
                actions = actions_by_env[env_idx]
                
                # Store observations and actions
                for agent_id in actions.keys():
                    rollouts_by_agent[agent_id]["obs"].append(self.observations[agent_id][env_idx])
                    rollouts_by_agent[agent_id]["actions"].append(actions[agent_id])
                
                # Step environment
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # Store rewards and dones
                for agent_id in actions.keys():
                    if agent_id in rewards:  # Agent might have been terminated
                        rollouts_by_agent[agent_id]["rewards"].append(rewards[agent_id])
                        rollouts_by_agent[agent_id]["dones"].append(
                            terminations.get(agent_id, False) or truncations.get(agent_id, False)
                        )
                    else:
                        # If agent was terminated, use zero reward and done=True
                        rollouts_by_agent[agent_id]["rewards"].append(0.0)
                        rollouts_by_agent[agent_id]["dones"].append(True)
                
                # Update observations for next step
                for agent_id in next_obs.keys():
                    if env_idx >= len(self.observations[agent_id]):
                        self.observations[agent_id].append(next_obs[agent_id])
                    else:
                        self.observations[agent_id][env_idx] = next_obs[agent_id]
                
                # Handle episode reset if all agents are done
                all_done = all(
                    terminations.get(agent_id, False) or truncations.get(agent_id, False)
                    for agent_id in actions.keys()
                )
                
                if all_done:
                    next_obs, _ = env.reset()
                    for agent_id in next_obs.keys():
                        self.observations[agent_id][env_idx] = next_obs[agent_id]
        
        # Compute advantages and returns for each agent
        for agent_id in rollouts_by_agent:
            # Convert lists to numpy arrays
            obs = np.array(rollouts_by_agent[agent_id]["obs"])
            rewards = np.array(rollouts_by_agent[agent_id]["rewards"])
            dones = np.array(rollouts_by_agent[agent_id]["dones"])
            
            # Compute values
            with torch.no_grad():
                values = []
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                # Get values in smaller batches to avoid OOM
                batch_size = 64
                for i in range(0, len(obs), batch_size):
                    batch_obs = obs_tensor[i:i+batch_size]
                    if self.policy.shared_policy:
                        batch_values = self.policy.agents["shared"].policy.get_value(batch_obs)
                    else:
                        batch_values = self.policy.agents[agent_id].policy.get_value(batch_obs)
                    values.append(batch_values.cpu().numpy())
                
                values = np.concatenate(values, axis=0).flatten()
            
            # Compute advantages and returns
            if self.policy.shared_policy:
                advantages, returns = self.policy.agents["shared"].compute_gae(
                    rewards, values, dones, values[-1]
                )
            else:
                advantages, returns = self.policy.agents[agent_id].compute_gae(
                    rewards, values, dones, values[-1]
                )
            
            rollouts_by_agent[agent_id]["values"] = values
            rollouts_by_agent[agent_id]["advantages"] = advantages
            rollouts_by_agent[agent_id]["returns"] = returns
            
            # Add placeholder for log_probs (will be computed during update)
            rollouts_by_agent[agent_id]["log_probs"] = np.zeros_like(rewards)
        
        return rollouts_by_agent
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def worker_process(args, rank, return_dict):
    try:
        # Set random seeds
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        
        # Create environment
        env_fn = lambda: make_cleanup_env(
            render_mode=None,
            num_agents=args.num_agents,
            max_cycles=2000,  # Adjust based on your needs
        )
        
        # Create temporary environment to get observation and action spaces
        temp_env = env_fn()
        obs, _ = temp_env.reset()
        
        # Get observation shape from the first agent's observation
        first_agent_id = list(obs.keys())[0]
        observation_shape = obs[first_agent_id].shape
        num_actions = temp_env.action_space(first_agent_id).n
        temp_env.close()
        
        # Create policy
        policy = PPOBaseline(
            observation_shape=observation_shape,
            num_actions=num_actions,
            num_agents=args.num_agents,
            lr=args.lr,
            entropy_coef=args.entropy_coeff,
            shared_policy=True,  # Use shared policy for all agents
            device="cuda" if args.gpus_per_worker > 0 else "cpu",
        )
        
        # Create rollout collector
        collector = RolloutCollector(
            env_fn=env_fn,
            policy=policy,
            num_envs=args.num_envs_per_worker,
            rollout_length=args.rollout_fragment_length,
        )
        
        # Collect rollouts
        rollouts = collector.collect()
        
        # Close collector
        collector.close()
        
        # Store rollouts in return_dict
        return_dict[rank] = rollouts
    except Exception as e:
        # Log the error for debugging
        print(f"Error in worker {rank}: {e}")
        return_dict[rank] = None

def parse_lr_schedule(steps_str, weights_str):
    """Parse learning rate schedule from command line arguments."""
    steps = [int(step) for step in steps_str.split()]
    weights = [float(weight) for weight in weights_str.split()]
    
    if len(steps) != len(weights):
        raise ValueError("Number of steps and weights must be equal")
    
    return list(zip(steps, weights))

def main():
    parser = argparse.ArgumentParser(description="Train PPO on Cleanup environment")
    
    # Add the arguments as per the provided command-line options
    parser.add_argument("--env", type=str, default="cleanup", help="Environment name")
    parser.add_argument("--model", type=str, default="baseline", help="Model name")
    parser.add_argument("--algorithm", type=str, default="PPO", help="Algorithm name")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of worker processes")
    parser.add_argument("--rollout_fragment_length", type=int, default=1000, help="Length of each rollout fragment")
    parser.add_argument("--num_envs_per_worker", type=int, default=16, help="Number of environments per worker")
    parser.add_argument("--stop_at_timesteps_total", type=int, default=500 * 10 ** 6, help="Stop condition (total timesteps)")
    parser.add_argument("--memory", type=int, default=160 * 10 ** 9, help="Memory limit (in bytes)")
    parser.add_argument("--cpus_per_worker", type=int, default=1, help="CPUs per worker")
    parser.add_argument("--gpus_per_worker", type=int, default=0, help="GPUs per worker")
    parser.add_argument("--gpus_for_driver", type=int, default=1, help="GPUs for driver")
    parser.add_argument("--cpus_for_driver", type=int, default=0, help="CPUs for driver")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of training samples")
    parser.add_argument("--entropy_coeff", type=float, default=0.00176, help="Entropy coefficient")
    parser.add_argument("--lr_schedule_steps", type=str, default="0 20000000", help="Learning rate schedule steps")
    parser.add_argument("--lr_schedule_weights", type=str, default=".00126 .000012", help="Learning rate schedule weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse the stop condition to get total timesteps
    # Expected format: "timesteps_total$((500*10**6))"
    
    # Parse learning rate schedule
    lr_schedule = parse_lr_schedule(args.lr_schedule_steps, args.lr_schedule_weights)
    args.lr = lr_schedule[0][1]  # Initial learning rate
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if args.gpus_for_driver > 0 else "cpu")
    
    # Create temporary environment to get observation and action spaces
    temp_env = make_cleanup_env(render_mode=None, num_agents=args.num_agents)

    reset_result = temp_env.reset()
    #print(f"Debug: temp_env.reset() returned {reset_result}")

    obs, _ = temp_env.reset()
    
    # Get observation shape from the first agent's observation
    first_agent_id = list(obs.keys())[0]
    observation_shape = obs[first_agent_id].shape

    #print(f"Debug: Observation shape for agent {first_agent_id}: {observation_shape}")
    #input()

    num_actions = temp_env.action_space(first_agent_id).n
    temp_env.close()
    
    # Create policy
    policy = PPOBaseline(
        observation_shape=observation_shape,
        num_actions=num_actions,
        num_agents=args.num_agents,
        lr=args.lr,
        entropy_coef=args.entropy_coeff,
        shared_policy=True,  # Use shared policy for all agents
        device=device,
    )
    
    # Training loop
    total_steps = 0
    max_steps = args.stop_at_timesteps_total
    update_interval = args.num_workers * args.num_envs_per_worker * args.rollout_fragment_length
    
    print(f"Starting training for {max_steps} steps")
    print(f"Update interval: {update_interval} steps")
    
    start_time = time.time()
    
    while total_steps < max_steps:
        # Update learning rate based on schedule
        for step, lr in lr_schedule:
            if total_steps >= step:
                for agent_name, agent in policy.agents.items():
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = lr
                break
        
        # Collect rollouts in parallel
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for rank in range(args.num_workers):
            p = mp.Process(target=worker_process, args=(args, rank, return_dict))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        for rank in range(args.num_workers):
            if rank not in return_dict or return_dict[rank] is None:
                print(f"Worker {rank} failed or did not return results. Check logs for details.")
                continue  # Skip this worker
        
        # Combine rollouts from all workers
        combined_rollouts = defaultdict(lambda: defaultdict(list))
        
        for rank in range(args.num_workers):
            worker_rollouts = return_dict[rank]
            for agent_id, agent_rollouts in worker_rollouts.items():
                for key, value in agent_rollouts.items():
                    combined_rollouts[agent_id][key].extend(value)
        
        # Convert lists to numpy arrays
        for agent_id in combined_rollouts:
            for key in combined_rollouts[agent_id]:
                combined_rollouts[agent_id][key] = np.array(combined_rollouts[agent_id][key])
        
        # Update policy
        update_metrics = policy.update(combined_rollouts)
        
        # Update step count
        steps_this_update = sum(len(combined_rollouts[agent_id]["rewards"]) for agent_id in combined_rollouts)
        total_steps += steps_this_update
        
        # Print progress
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / elapsed
        remaining = (max_steps - total_steps) / steps_per_sec if steps_per_sec > 0 else 0
        
        print(f"Steps: {total_steps}/{max_steps} ({100*total_steps/max_steps:.1f}%), "f"SPS: {steps_per_sec:.1f}, ETA: {remaining/60:.1f}m")
        
        # Print metrics
        if update_metrics:
            if "shared" in update_metrics:
                metrics = update_metrics["shared"]
                print(f"Loss: {metrics['total_loss']:.4f}, "
                      f"Policy Loss: {metrics['policy_loss']:.4f}, "
                      f"Value Loss: {metrics['value_loss']:.4f}, "
                      f"Entropy: {metrics['entropy']:.4f}")
            else:
                # Print metrics for first agent only to avoid clutter
                first_agent = list(update_metrics.keys())[0]
                metrics = update_metrics[first_agent]
                print(f"Agent {first_agent} - Loss: {metrics['total_loss']:.4f}, "
                      f"Policy Loss: {metrics['policy_loss']:.4f}, "
                      f"Value Loss: {metrics['value_loss']:.4f}, "
                      f"Entropy: {metrics['entropy']:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_{total_steps}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        policy.save(os.path.join(checkpoint_dir, "model"))
        
        # Save latest checkpoint
        policy.save(os.path.join(args.output_dir, "model_latest"))

if __name__ == "__main__":
    main()