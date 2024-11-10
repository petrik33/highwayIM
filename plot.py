# PLOT TRAINING METRICS FROM MONITOR LOG
import os
import numpy as np
import matplotlib.pyplot as plt

log_dir = "./logs/"
plot_dir = "./plots"
tensorboard_log_dir = "./ppo_tensorboard/"
model_dir = "./model"
model_name = "ppo_intersection"
model_version = "v1"

# Load training data from the Monitor log CSV file
monitor_data = np.genfromtxt(os.path.join(log_dir, f"monitor_train_log_{model_version}.csv.monitor.csv"), delimiter=',', skip_header=1)

# Extract episode rewards and lengths
timesteps = monitor_data[:, 0]  # Timesteps of each episode
episode_rewards = monitor_data[:, 1]  # Episode rewards
episode_lengths = monitor_data[:, 2]  # Episode lengths

# Plot episode rewards over time
plt.figure(figsize=(12, 6))
plt.plot(timesteps, episode_rewards, label='Episode Rewards')
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.title('Training Rewards Over Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"rewards_plot_{model_version}.png"))  # Save reward plot

# Plot episode lengths over time
plt.figure(figsize=(12, 6))
plt.plot(timesteps, episode_lengths, label='Episode Lengths', color='orange')
plt.xlabel('Timesteps')
plt.ylabel('Episode Lengths')
plt.title('Episode Lengths Over Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"lengths_plot_{model_version}.png"))  # Save episode length plot
