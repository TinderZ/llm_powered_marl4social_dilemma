# run_scripts/train_rllib.py
import argparse
import os
import sys
from datetime import datetime
import pytz

import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig # Using PPO directly
from ray.rllib.policy.policy import PolicySpec # <--- 新增导入

# Add near the top with other imports
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time # Optional for smoother plotting in some backends
from ray.tune.callback import Callback
# import threading # To handle plotting in a separate thread potentially
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Import your refactored environment and model ---
# Assuming PettingZoo AEC interface for the environment creator
from envs.cleanup_env import env as cleanup_env_creator
# Import other env creators if needed (e.g., harvest)
# Import your refactored model(s)
from models.baseline_model import BaselineModel
# from models.moa_model import MOAModel  # Example if extending
# from models.scm_model import SocialCuriosityModule # Example if extending



class PlottingCallback(Callback):
    """
    A Tune Callback that plots agent metrics (loss, reward, reward variance)
    during training and saves the plot periodically.
    """
    def __init__(self, num_agents, plot_freq=5, save_path="training_metrics.png"):
        super().__init__()
        self.num_agents = num_agents
        self.plot_freq = plot_freq # Plot every N iterations
        # self.save_path = save_path
        self._iter = 0

        # Data storage: dictionary mapping agent_id to list of metrics
        self.policy_loss = defaultdict(list)
        self.mean_reward = defaultdict(list) # Use mean reward as proxy for total
        self.reward_variance = defaultdict(list) # Store overall variance here for simplicity
        self.timesteps = [] # Shared x-axis

        # Plotting setup (run in main thread initially)
        self._setup_plot()
        # self._lock = threading.Lock() # Lock for thread safety if needed

    def _setup_plot(self):
        """Initialize the Matplotlib figure and axes."""
        # plt.ion() # Turn on interactive mode if needed, but Agg backend is better for saving
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        self.fig.suptitle("Agent Training Metrics")

        self.axes[0].set_ylabel("Mean Policy Loss")
        self.axes[1].set_ylabel("Mean Episode Reward")
        self.axes[2].set_ylabel("Episode Reward Variance (Overall)")
        self.axes[2].set_xlabel("Training Timesteps")

        # Initial empty plot lines for legends
        self.lines_loss = {}
        self.lines_reward = {}
        self.line_variance = None # Only one line for overall variance

        agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_agents)) # Use a colormap

        for i, agent_id in enumerate(agent_ids):
            self.lines_loss[agent_id], = self.axes[0].plot([], [], label=f"{agent_id} Loss", color=colors[i])
            self.lines_reward[agent_id], = self.axes[1].plot([], [], label=f"{agent_id} Reward", color=colors[i])

        # Use a single line for overall reward variance across all agents
        self.line_variance, = self.axes[2].plot([], [], label="Overall Reward Variance", color='red')

        self.axes[0].legend(loc='upper right')
        self.axes[1].legend(loc='lower right')
        self.axes[2].legend(loc='upper right')
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap


    def on_trial_result(self, iteration: int, trials: list, trial: "Trial", result: dict, **info):
        """Called after each training iteration for a trial."""
        self._iter += 1
        if self._iter % self.plot_freq != 0:
            return # Only plot every N iterations
        
        if not trial.logdir:
             # Should not happen typically after trial starts, but check just in case
             print(f"PlottingCallback: Warning - trial.logdir not available in iteration {self._iter}. Skipping plot save.")
             return
        # Define the filename within the trial directory
        save_path = os.path.join(trial.logdir, "training_metrics.png") # <-- DYNAMIC PATH


        current_timesteps = result.get("timesteps_total", self._iter) # Use timesteps if available
        self.timesteps.append(current_timesteps)

        # --- Data Extraction ---
        # Policy Loss (handle potential variations in result structure)
        policy_losses_found = False
        if "info" in result and "learner" in result["info"]:
            for agent_id, policy_info in result["info"]["learner"].items():
                 # Check if agent_id matches expected format and learner_stats exists
                if agent_id.startswith("agent_") and "learner_stats" in policy_info:
                    loss = policy_info["learner_stats"].get("policy_loss")
                    if loss is not None:
                         self.policy_loss[agent_id].append(loss)
                         policy_losses_found = True
                    else:
                         # Append NaN or previous value if loss is missing for this step
                         self.policy_loss[agent_id].append(self.policy_loss[agent_id][-1] if self.policy_loss[agent_id] else np.nan)

        # Fallback or if structure is different (might be aggregated)
        if not policy_losses_found and "policy_loss" in result:
             # This is likely aggregated, plot the same for all agents
             agg_loss = result["policy_loss"]
             for i in range(self.num_agents):
                agent_id = f"agent_{i}"
                self.policy_loss[agent_id].append(agg_loss)


        # Mean Episode Reward per Policy
        rewards_found = False
        if "policy_reward_mean" in result and isinstance(result["policy_reward_mean"], dict):
            for agent_id, reward in result["policy_reward_mean"].items():
                if agent_id.startswith("agent_"):
                    self.mean_reward[agent_id].append(reward)
                    rewards_found = True
            # Fill missing agents for this step if needed
            for i in range(self.num_agents):
                agent_id = f"agent_{i}"
                if agent_id not in result["policy_reward_mean"]:
                    self.mean_reward[agent_id].append(self.mean_reward[agent_id][-1] if self.mean_reward[agent_id] else np.nan)

        # Fallback using overall mean reward
        if not rewards_found and "episode_reward_mean" in result:
            agg_reward = result["episode_reward_mean"]
            for i in range(self.num_agents):
                agent_id = f"agent_{i}"
                self.mean_reward[agent_id].append(agg_reward)


        # Reward Variance (using overall episode rewards)
        variance = np.nan # Default to NaN
        if "hist_stats" in result and "episode_reward" in result["hist_stats"]:
            episode_rewards = result["hist_stats"]["episode_reward"]
            if len(episode_rewards) > 1: # Need at least 2 points for variance
                variance = np.var(episode_rewards)
        # Append the same overall variance to the shared list
        self.reward_variance["overall"].append(variance)


        # --- Update Plot ---
        # Use lock for potential threading issues, though Agg backend might avoid them
        agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        min_len = len(self.timesteps) # Ensure all data lists match length of x-axis

        for agent_id in agent_ids:
            # Pad data if necessary (e.g., if an agent's data was missing initially)
            while len(self.policy_loss[agent_id]) < min_len:
                self.policy_loss[agent_id].insert(0, np.nan) # Pad start
            while len(self.mean_reward[agent_id]) < min_len:
                self.mean_reward[agent_id].insert(0, np.nan) # Pad start

            # Update plot data if lines exist
            if agent_id in self.lines_loss:
                self.lines_loss[agent_id].set_data(self.timesteps, self.policy_loss[agent_id][-min_len:])
            if agent_id in self.lines_reward:
                self.lines_reward[agent_id].set_data(self.timesteps, self.mean_reward[agent_id][-min_len:])

        # Update overall variance plot data
        while len(self.reward_variance["overall"]) < min_len:
            self.reward_variance["overall"].insert(0, np.nan)
        # Update variance line if it exists
        if self.line_variance:
            self.line_variance.set_data(self.timesteps, self.reward_variance["overall"][-min_len:])
        # Rescale axes
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(True, True, True)

        # Redraw and save
        try:
            self.fig.canvas.draw_idle() # Request redraw
            self.fig.savefig(save_path) # <-- Use the dynamic save_path
            # print(f"PlottingCallback: Saved plot to {save_path}") # Optional: for debugging
        except Exception as e:
             # Log error but don't crash the whole training run
             print(f"PlottingCallback: Failed to save plot to {save_path}: {e}")
    # plt.pause(0.01) # Small pause might be needed for some backends/interactive use

            # plt.pause(0.01) # Small pause might be needed for some backends/interactive use

    # Optional: Close plot when trial ends or experiment finishes
    # def on_trial_complete(self, iteration: int, trials: list, trial: "Trial", **info):
    #     self.close_plot()

    # def on_experiment_end(self, trials: list, **info):
    #     self.close_plot()

    def close_plot(self):
         with self._lock:
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
                self.fig = None
                self.axes = None
                # plt.ioff() # Turn off interactive mode if it was turned on




# --- Environment Registration ---
# It's common to register environments here before Tune runs.
def env_creator(env_config):
    env_name = env_config.get("env_name", "cleanup")
    num_agents = env_config.get("num_agents", 2)
    # ... other env args ...

    if env_name == "cleanup":
        # Create the PettingZoo AEC environment first
        aec_env = cleanup_env_creator(num_agents=num_agents) # , use_llm=use_llm, etc.
        # Wrap it with RLlib's wrapper
        rllib_multi_agent_env = PettingZooEnv(aec_env)
        return rllib_multi_agent_env
    # elif env_name == "harvest":
        # aec_env = harvest_aec_creator(num_agents=num_agents)
        # return PettingZooEnv(aec_env)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
# Register the environment creator function under a unique name
ENV_NAME_REGISTERED = "ssd_cleanup_v1" # Or make this dynamic based on args.env
register_env(ENV_NAME_REGISTERED, env_creator)


# --- Model Registration ---
# Register custom models with RLlib
# You can choose unique names or use the class directly in config
ModelCatalog.register_custom_model("baseline_model_refactored", BaselineModel)
# ModelCatalog.register_custom_model("moa_model_refactored", MOAModel) # Example
# ModelCatalog.register_custom_model("scm_model_refactored", SocialCuriosityModule) # Example


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment Identification
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name prefix."
    )
    parser.add_argument(
        "--env", type=str, default="cleanup", choices=["cleanup", "harvest"], # Add others if refactored
        help="Environment name."
    )
    parser.add_argument(
        "--algorithm", type=str, default="PPO", choices=["PPO"], # Extend if needed
        help="RLlib algorithm."
    )
    parser.add_argument(
        "--model", type=str, default="baseline", choices=["baseline", "moa", "scm"], # Add others if refactored
        help="Model architecture."
    )
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of trials to run.")
    parser.add_argument( "--seed", type=int, default=None, help="Set seed for reproducibility.")


    # Ray and Tune Control
    parser.add_argument("--local_mode", action="store_true", help="Run Ray in local mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if found.")
    parser.add_argument("--restore", type=str, default=None, help="Explicit path to checkpoint to restore from.")
    parser.add_argument("--use_s3", action="store_true", help="Upload results to S3.")
    parser.add_argument("--s3_bucket_prefix", type=str, default="s3://your-bucket-name/ssd-results", help="S3 bucket prefix for uploads.") # CHANGE BUCKET NAME
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Save checkpoint every N iterations.")
    parser.add_argument("--stop_timesteps", type=int, default=int(500e6), help="Stop after N total env steps.")
    parser.add_argument("--stop_reward", type=float, default=None, help="Stop if avg reward reaches this value.")
    parser.add_argument("--stop_iters", type=int, default=None, help="Stop after N training iterations.")


    # Resource Allocation (match run script)
    parser.add_argument("--num_workers", type=int, default=6, help="Number of rollout workers.")
    parser.add_argument("--num_envs_per_worker", type=int, default=16, help="Number of envs per worker.")
    parser.add_argument("--cpus_per_worker", type=float, default=1, help="CPUs per worker.")
    parser.add_argument("--gpus_per_worker", type=float, default=0, help="GPUs per worker.")
    parser.add_argument("--cpus_for_driver", type=int, default=1, help="CPUs for the driver (trainer).")
    parser.add_argument("--gpus_for_driver", type=float, default=1, help="GPUs for the driver (trainer).")

    # Core PPO Hyperparameters (match run script)
    parser.add_argument("--rollout_fragment_length", type=int, default=1000, help="RLlib rollout fragment length.")
    parser.add_argument("--train_batch_size", type=int, default=None, help="RLlib train batch size (if None, calculated).")
    parser.add_argument("--sgd_minibatch_size", type=int, default=None, help="RLlib SGD minibatch size (if None, calculated).")
    parser.add_argument("--num_sgd_iter", type=int, default=10, help="Number of SGD iterations per train batch.") # Default PPO is 10-30
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides schedule if set).")
    parser.add_argument("--lr_schedule_steps", nargs="+", type=int, default=[0, 20000000], help="Timesteps for LR schedule points.")
    parser.add_argument("--lr_schedule_weights", nargs="+", type=float, default=[0.00126, 0.000012], help="LR values for schedule points.")
    parser.add_argument("--entropy_coeff", type=float, default=0.00176, help="Entropy coefficient.")
    parser.add_argument("--vf_loss_coeff", type=float, default=0.5, help="Value function loss coefficient (PPO default).") # Common PPO default
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter (PPO default).") # Common PPO default
    parser.add_argument("--grad_clip", type=float, default=40.0, help="Gradient clipping.")

    # Model Hyperparameters (Specific to your refactored models)
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="LSTM hidden state size.")
    # Add other model-specific args if they differ from defaults in BaselineModel etc.
    # e.g., parser.add_argument("--fcnet_hiddens", nargs='+', type=int, default=[32, 32])

    # Environment Specific Args (if needed)
    parser.add_argument("--use_collective_reward", action="store_true", help="Use collective reward.")
    # Add other env args if they influence the env_creator

    # LLM Args (if applicable)
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM features in the environment.")
    parser.add_argument("--llm_f_step", type=int, default=50, help="LLM update frequency in steps.")

    args = parser.parse_args()

    # Calculate default batch sizes if not provided
    if args.train_batch_size is None:
        args.train_batch_size = args.num_workers * args.num_envs_per_worker * args.rollout_fragment_length
        print(f"Calculated train_batch_size: {args.train_batch_size}")
    if args.sgd_minibatch_size is None:
        # PPO often uses smaller minibatches than the full train batch
        # A common default is 128 or 256, or derived from train_batch_size
        args.sgd_minibatch_size = max(128, args.train_batch_size // 16) # Example derivation
        print(f"Calculated sgd_minibatch_size: {args.sgd_minibatch_size}")

    return args


# --- Main Execution ---
def main(args):
    # Initialize Ray
    if args.local_mode:
        ray.init(num_cpus=args.cpus_for_driver + args.num_workers * args.cpus_per_worker,
                 local_mode=True)
    else:
        # Connect to existing cluster or start new one
        ray.init(address=os.environ.get("RAY_ADDRESS", None)) # Assumes RAY_ADDRESS is set for clusters


    temp_env_config = {
        "env_name": args.env,
        "num_agents": args.num_agents,
        "use_llm": args.use_llm,
        "llm_f_step": args.llm_f_step,
    }
    temp_env = env_creator(temp_env_config)
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()


    
    # --- Configure Algorithm ---
    if args.algorithm == "PPO":
        config = PPOConfig()
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    # Select correct registered model name
    if args.model == "baseline":
        model_name_registered = "baseline_model_refactored"
    # elif args.model == "moa":
    #     model_name_registered = "moa_model_refactored"
    # elif args.model == "scm":
    #     model_name_registered = "scm_model_refactored"
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Learning Rate Schedule
    if args.lr is not None:
        lr_schedule = None # Override schedule if fixed LR is given
        lr_value = args.lr
    elif args.lr_schedule_steps and args.lr_schedule_weights:
        lr_schedule = list(zip(args.lr_schedule_steps, args.lr_schedule_weights))
        lr_value = args.lr_schedule_weights[0] # Initial LR
    else:
        lr_schedule = None
        lr_value = 0.0001 # Default if nothing specified

    # Environment Config (passed to env_creator)
    env_config = {
        "env_name": args.env,
        "num_agents": args.num_agents,
        "use_llm": args.use_llm,
        "llm_f_step": args.llm_f_step,
        # Add other env-specific args here if needed by creator
    }

    config = (
        config
        .environment(
            env=ENV_NAME_REGISTERED,
            env_config=env_config,
            disable_env_checking=True # Recommended for multi-agent/complex envs
        )
        .framework("torch") # Or "tf2"
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length
        )
        .training(
            gamma=0.99,
            lr=lr_value,
            lr_schedule=lr_schedule,
            lambda_=0.95, # GAE lambda (PPO default)
            kl_coeff=0.2, # PPO default
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            train_batch_size=args.train_batch_size,
            vf_loss_coeff=args.vf_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            clip_param=args.clip_param,
            grad_clip=args.grad_clip,
            model={
                "custom_model": model_name_registered,
                "custom_model_config": {
                    # Pass model-specific args from command line or defaults
                    # Ensure these match your refactored model's __init__
                    "conv_filters": [[6, [3, 3], 1]], # Example default
                    "fcnet_hiddens": [32, 32], # Example default
                    "lstm_hidden_size": args.lstm_hidden_size,
                },
                 "use_lstm": False, # Let RLlib handle if model is nn.Module? See BaselineModel notes.
            },
        )
        # .multi_agent(
        #     # Assuming identical policies for all agents using the same model class
        #     policies={f"agent_{i}" for i in range(args.num_agents)},
        #      # Map agent_id "agent_0", "agent_1", etc. to policy_id "agent_0", "agent_1", etc.
        #      # policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: f"agent_{agent_id.split('_')[-1]}"),
        #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id), # Simpler mapping if policy keys match agent ids
        # )
        .multi_agent(
            policies={
                f"agent_{i}": PolicySpec(
                    policy_class=None,  # 让 RLlib 使用默认的 PPO TorchPolicy
                    observation_space=obs_space, # 显式提供观察空间
                    action_space=act_space,      # 显式提供动作空间
                    # config 可以省略，让其继承顶层 config 的 model 设置
                    # 或者如果需要为特定策略覆盖配置，可以在这里添加:
                    # config={"model": {... specific overrides ...}}
                )
                for i in range(args.num_agents)
            },
            # 保持策略映射不变，将 agent_id 映射到对应的 policy_id
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(
            num_gpus=args.gpus_for_driver,
            num_cpus_per_worker=args.cpus_per_worker,
            num_gpus_per_worker=args.gpus_per_worker,
            num_cpus_for_local_worker = args.cpus_for_driver, # Renamed from driver
        )
         # Add evaluation config if needed
         #.evaluation(evaluation_interval=10, evaluation_num_workers=1)
         .debugging(seed=args.seed) # Set seed if provided, None otherwise # Set seed if provided
    )


    # --- Stopping Criteria ---
    stop_criteria = {}
    if args.stop_timesteps:
        stop_criteria["timesteps_total"] = args.stop_timesteps
    if args.stop_reward:
        stop_criteria["episode_reward_mean"] = args.stop_reward
    if args.stop_iters:
        stop_criteria["training_iteration"] = args.stop_iters
    if not stop_criteria:
        stop_criteria["training_iteration"] = 100 # Default stop after 100 iters if nothing else set


    # --- Experiment Naming and Storage ---
    experiment_base_name = args.exp_name if args.exp_name else f"{args.env}_{args.model}_{args.algorithm}"
    # Add date/time for uniqueness?
    # timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
    # experiment_full_name = f"{experiment_base_name}_{timestamp}"
    experiment_full_name = experiment_base_name # Keep it simple for now

    storage_path = os.path.expanduser("~/ray_results")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    if args.use_s3:
        # Ensure path ends with / for S3 uploads
        s3_prefix = args.s3_bucket_prefix
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'
        storage_path = s3_prefix

    plot_callback = PlottingCallback(
        num_agents=args.num_agents,
        plot_freq=5,  # Update plot every 5 training iterations (adjust as needed)
        # save_path=os.path.join(os.path.expanduser("ray_results"), # Save in default results dir
        #                         f"{experiment_full_name}_metrics.png") # Filename based on experiment
    )

    # --- Setup Tune ---
    tuner = tune.Tuner(
        args.algorithm, # Trainable name (e.g., "PPO")
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            name=experiment_full_name,
            stop=stop_criteria,
            storage_path=storage_path, # Local path or S3 prefix
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
                num_to_keep=3 # Keep last 3 checkpoints
            ),
            callbacks=[plot_callback]
            # Add failure config if needed
            # failure_config=ray.air.FailureConfig(max_failures=-1), # Infinite retries
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples, # Number of trials
            # Add scheduler if doing hyperparameter tuning
            # scheduler=pbt_scheduler,
            metric="episode_reward_mean", # Metric to optimize (if tuning)
            mode="max", # Optimization mode (if tuning)
        ),
    )

    # --- Restore and Run ---
    if args.resume:
         print(f"Attempting to resume experiment: {experiment_full_name} from {storage_path}")
         # Note: Tuner automatically handles resuming if the experiment name/path exists
         # tuner = tune.Tuner.restore(os.path.join(storage_path, experiment_full_name), trainable=args.algorithm)
         # The above might be needed for specific resume cases, but Tuner(..., run_config=...) often handles it.
         pass # Tuner handles resume based on name/path
    elif args.restore:
         print(f"Restoring experiment from checkpoint: {args.restore}")
         # Restore requires the specific trainable and path to checkpoint *directory*
         tuner = tune.Tuner.restore(path=args.restore, trainable=args.algorithm)
         # Need to potentially re-apply some config/stop criteria if not in checkpoint?
         # tuner.update_config(...) # Less common, usually restore loads most things

    # Run the experiment(s)
    results = tuner.fit()

    print("Training finished.")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # --- 修改开始: 增加检查 ---
    if best_result:
        print("Best trial config: {}".format(best_result.config))
        if best_result.metrics and "episode_reward_mean" in best_result.metrics:
            print("Best trial final reward: {}".format(best_result.metrics["episode_reward_mean"]))
        else:
            print("Best trial found, but 'episode_reward_mean' metric is missing.")
            print(f"All metrics for best trial: {best_result.metrics}")
    else:
        print("No best trial found (likely due to errors or no completed trials).")
    # --- 修改结束 ---

    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    # Handle potential debug mode setting local_mode
    if sys.gettrace() is not None:
         print("Debug mode detected, forcing local_mode=True")
         args.local_mode = True
         if args.exp_name is None:
             args.exp_name = "debug_experiment" # Override name for debug runs

    main(args)