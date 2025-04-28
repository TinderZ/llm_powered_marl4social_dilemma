#!/bin/bash

# Set experiment name
EXP_NAME="cleanup_baseline_ppo_refactored"

# Set environment and model configurations
ENV="cleanup"
MODEL="baseline"
ALGORITHM="PPO"

# Set hyperparameters
NUM_AGENTS=5
NUM_WORKERS=3  # Optimized for multi-GPU parallelism
NUM_ENVS_PER_WORKER=8  # Increased parallelism for environment instances
ROLL_OUT_FRAGMENT_LENGTH=1000
STOP_TIMESTEPS=500000000
ENTROPY_COEFF=0.00176
LR_SCHEDULE_STEPS="0 20000000"
LR_SCHEDULE_WEIGHTS="0.00126 0.000012"
GRAD_CLIP=40.0

# Set GPU configuration
CPUS_PER_WORKER=4  # Adjusted based on CPU core availability
GPUS_PER_WORKER=1  # Each worker uses one GPU
CPUS_FOR_DRIVER=4  # Number of CPU cores available for driver
GPUS_FOR_DRIVER=1  # Use one GPU for driver (since GPUs are powerful)

# Set up Ray configuration
export RAY_MEMORY=160000000000  # Example memory allocation for Ray workers
export RAY_ALLOW_MULTI_GPU=1  # Enable multi-GPU mode for Ray

# Run training with optimizations for multi-GPU setup
python train_rllib.py \
  --exp_name $EXP_NAME \
  --env $ENV \
  --model $MODEL \
  --algorithm $ALGORITHM \
  --num_agents $NUM_AGENTS \
  --num_workers $NUM_WORKERS \
  --num_envs_per_worker $NUM_ENVS_PER_WORKER \
  --rollout_fragment_length $ROLL_OUT_FRAGMENT_LENGTH \
  --stop_timesteps $STOP_TIMESTEPS \
  --cpus_per_worker $CPUS_PER_WORKER \
  --gpus_per_worker $GPUS_PER_WORKER \
  --cpus_for_driver $CPUS_FOR_DRIVER \
  --gpus_for_driver $GPUS_FOR_DRIVER \
  --entropy_coeff $ENTROPY_COEFF \
  --lr_schedule_steps $LR_SCHEDULE_STEPS \
  --lr_schedule_weights $LR_SCHEDULE_WEIGHTS \
  --grad_clip $GRAD_CLIP
