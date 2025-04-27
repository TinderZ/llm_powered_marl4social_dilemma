# algorithms/ppo_baseline_rllib.py
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from models.baseline_model import BaselineModel # Assuming refactored model path
# Assuming refactored env path and creator function exists
from envs.cleanup_env import env as cleanup_env_creator
# import gymnasium as gym # If needed for spaces

# Note: Environment registration might happen in the main training script instead,
# but it's shown here for clarity if you want modular registration.
# def env_creator(env_config):
#     # Assuming your refactored env follows PettingZoo API
#     # You might need to wrap it if it's a raw ParallelEnv
#     # Example for cleanup:
#     return cleanup_env_creator(num_agents=env_config.get("num_agents", 2))

# ENV_NAME = "cleanup_v1_refactored" # Choose a unique name
# register_env(ENV_NAME, env_creator)

def get_ppo_baseline_config(num_agents: int, env_name: str) -> PPOConfig:
    """
    Creates an RLlib PPO AlgorithmConfig for the baseline model on the specified environment.

    Args:
        num_agents: Number of agents in the environment.
        env_name: Registered name of the environment to use (e.g., "cleanup_v1_refactored").

    Returns:
        An RLlib PPOConfig object.
    """

    # Example: Define observation and action spaces directly if needed,
    # or let RLlib infer them from the registered environment.
    # Assuming RLlib infers spaces from the registered PettingZoo env.
    # If not, you might need:
    # dummy_env = env_creator({"num_agents": num_agents})
    # obs_space = dummy_env.observation_space("agent_0") # Get space for one agent
    # act_space = dummy_env.action_space("agent_0")
    # dummy_env.close()

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True) # Use the registered env name
        .framework("torch") # Or "tf2"
        .rollouts(num_rollout_workers=1) # Default, adjust in run script
        .training(
            model={
                "custom_model": BaselineModel, # Register or reference directly
                 # --- Pass necessary model config ---
                 # These should match the constructor of your refactored BaselineModel
                 "custom_model_config": {
                     # Example values - Adjust based on your refactored model's needs
                     # Assuming BaselineModel takes these args, see models/baseline_model.py
                     "conv_filters": [[6, [3, 3], 1]], # Example from original
                     "fcnet_hiddens": [32, 32],     # Example from original
                     "lstm_hidden_size": 128,       # Example from original
                     # Add any other args your refactored BaselineModel expects
                 },
                 "use_lstm": False, # RLlib handles recurrence automatically with custom models usually
                                    # If your BaselineModel *doesn't* handle seq internally, set True
                                    # and ensure it adheres to RLlib's RecurrentNetwork API.
                                    # But typically, if model is nn.Module, RLlib wraps it.
                 # "lstm_cell_size": 128, # Often handled by custom_model_config now
            },
            # Other training params (lr, gamma, etc.) can be set here or in run script
            # gamma=0.99,
            # lr=0.0001,
            # train_batch_size=4000, # Example
        )
        .multi_agent(
            # Policies are identical instances of the same BaselineModel
            policies={f"agent_{i}" for i in range(num_agents)},
            # Map all agent IDs to the single policy defined above
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: f"agent_{agent_id.split('_')[-1]}"),
            # Or if policies are truly identical and parameters shared:
            # policies = {"shared_policy": PolicySpec(policy_class=None, # Inferred by default
            #                                         observation_space=obs_space, # Provide if not inferred
            #                                         action_space=act_space,      # Provide if not inferred
            #                                         config={"custom_model": BaselineModel, ...})}
            # policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"

        )
        # Add resource configuration if needed, often done in run script
        # .resources(num_gpus=0)
    )

    return config

# Example usage (optional, for testing this file)
# if __name__ == "__main__":
#     NUM_TEST_AGENTS = 2
#     config = get_ppo_baseline_config(NUM_TEST_AGENTS, ENV_NAME)
#     print("Generated PPO Config:")
#     print(config.to_dict())

#     # Example of building the algorithm (requires Ray to be initialized)
#     # import ray
#     # ray.init(local_mode=True)
#     # algo = config.build()
#     # print("Algorithm built successfully.")
#     # ray.shutdown()