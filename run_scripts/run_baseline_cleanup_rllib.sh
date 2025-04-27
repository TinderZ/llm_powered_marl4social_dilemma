#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Construct the path to the python script relative to this script
TRAIN_SCRIPT_PATH="$SCRIPT_DIR/train_rllib.py"

# Check if the training script exists
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT_PATH"
    exit 1
fi

# Experiment name prefix (optional, train_rllib.py creates default if None)
EXP_NAME="cleanup_baseline_ppo_refactored"

# Run the refactored training script
python "$TRAIN_SCRIPT_PATH" \
--exp_name "$EXP_NAME" \
--env cleanup \
--model baseline \
--algorithm PPO \
--num_agents 5 \
--num_workers 6 \
--num_envs_per_worker 16 \
--rollout_fragment_length 1000 \
--stop_timesteps $((500 * 1000 * 1000)) \
`# Resource allocation (Important: Ensure these align with your Ray cluster setup)` \
`# Ray defaults usually work well if starting Ray on a single machine.` \
`# Explicitly set if running on a cluster or needing specific allocation.` \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--cpus_for_driver 1 \
--gpus_for_driver 1 \
`# Hyperparameters (matching original where possible)` \
--entropy_coeff 0.00176 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.00126 0.000012 \
--grad_clip 40.0 \
`# Add other relevant args from train_rllib.py if needed, e.g.:` \
`# --sgd_minibatch_size 2048` \
`# --num_sgd_iter 10` \
`# --vf_loss_coeff 0.5` \
`# --clip_param 0.2` \
`# --num_samples 5` \
`# --checkpoint_freq 100` \
"$@"

# "$@" passes any additional command line arguments to the script