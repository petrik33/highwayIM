# EVALUATE AND VISUALIZE MODEL PERFORMANCE
import highway_env
import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

log_dir = "./logs/"
model_dir = "./model"
model_name = "a2c_intersection"
model_version = "v2"

env = gym.make('intersection-v0', render_mode="human")
env = Monitor(env, filename=os.path.join(log_dir, f"monitor_test_log_{model_version}.csv"))

# Load the trained model for evaluation
trained_model = PPO.load(os.path.join(model_dir, f"{model_name}_{model_version}"))

# Evaluate the trained model over 10 episodes
mean_reward, std_reward = evaluate_policy(trained_model, env, n_eval_episodes=20, render=True)

print(f"Evaluation: Mean reward = {mean_reward}, Std reward = {std_reward}")
