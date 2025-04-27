# run_scripts/train_rllib.py
import argparse
import os
import sys
from datetime import datetime
import pytz

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig # Using PPO directly

# --- Import your refactored environment and model ---
# Assuming PettingZoo AEC interface for the environment creator
from envs.cleanup_env import env as cleanup_env_creator
# Import other env creators if needed (e.g., harvest)

# Import your refactored model(s)
from models.baseline_model import BaselineModel
# from models.moa_model import MOAModel  # Example if extending
# from models.scm_model import SocialCuriosityModule # Example if extending


# --- Environment Registration ---
# It's common to register environments here before Tune runs.
def env_creator(env_config):
    env_name = env_config.get("env_name", "cleanup") # Default to cleanup
    num_agents = env_config.get("num_agents", 2)
    # Add other env-specific configs if needed
    # use_llm = env_config.get("use_llm", False)
    # llm_f_step = env_config.get("llm_f_step", 50)

    if env_name == "cleanup":
        # Pass num_agents and potentially other config to your creator
        return cleanup_env_creator(num_agents=num_agents) #, use_llm=use_llm, llm_f_step=llm_f_step)
    # elif env_name == "harvest":
        # return harvest_env_creator(num_agents=num_agents)
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
        .multi_agent(
            # Assuming identical policies for all agents using the same model class
            policies={f"agent_{i}" for i in range(args.num_agents)},
             # Map agent_id "agent_0", "agent_1", etc. to policy_id "agent_0", "agent_1", etc.
             # policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: f"agent_{agent_id.split('_')[-1]}"),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id), # Simpler mapping if policy keys match agent ids
        )
        .resources(
            num_gpus=args.gpus_for_driver,
            num_cpus_per_worker=args.cpus_per_worker,
            num_gpus_per_worker=args.gpus_per_worker,
            num_cpus_for_local_worker = args.cpus_for_driver, # Renamed from driver
        )
         # Add evaluation config if needed
         #.evaluation(evaluation_interval=10, evaluation_num_workers=1)
         .debugging(seed=args.seed if args.seed is not None else -1) # Set seed if provided
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


    storage_path = os.path.expanduser("~/ray_results") # Default Ray results directory
    if args.use_s3:
        # Ensure path ends with / for S3 uploads
        s3_prefix = args.s3_bucket_prefix
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'
        storage_path = s3_prefix


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
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final reward: {}".format(best_result.metrics["episode_reward_mean"]))

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