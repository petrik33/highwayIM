import gymnasium as gym
import highway_env
import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.logger import configure
import pprint
import tqdm
import rich

log_dir = "./logs/"
model_dir = "./model"
model_name = "a2c_intersection"
model_version = "v2"

# 1. SETUP LOGGING AND ENVIRONMENT

# Create log directories
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment with Monitor for logging
env = gym.make('intersection-v0')
env = Monitor(env, filename=os.path.join(log_dir, "monitor_train_log.csv"))

# Configure Tensorboard logger
tensorboard_log_dir = "./ppo_tensorboard/"
os.makedirs(tensorboard_log_dir, exist_ok=True)
new_logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])

# 2. INSTANTIATE THE MODEL

# PPO Model initialization with the environment and Tensorboard log
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
            gamma=0.85, ent_coef=0.01)
model.set_logger(new_logger)

# 3. CALLBACKS FOR CHECKPOINTING AND EVALUATION DURING TRAINING

# Create checkpoint callback to save the model at regular intervals
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=model_dir, name_prefix=f"{model_name}_{model_version}")

# Create evaluation callback to evaluate the model during training
eval_env = gym.make('intersection-v0')
eval_env = Monitor(eval_env, filename=os.path.join(log_dir, f"monitor_eval_log_{model_version}.csv"))
eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir, log_path=log_dir, eval_freq=10000)

progress_bar_callback = ProgressBarCallback()

# Combine both callbacks into a list
callback_list = CallbackList([checkpoint_callback, eval_callback, progress_bar_callback])

# 4. TRAIN THE MODEL

# Train the PPO model for 100,000 timesteps, using callbacks for checkpointing and evaluation
model.learn(total_timesteps=100000, callback=callback_list)

# Save the trained model
model.save(os.path.join(model_dir, f"{model_name}_{model_version}"))

# 5. CLEANUP

env.close()

print("Training complete. The model, logs, and visualizations have been saved.")
