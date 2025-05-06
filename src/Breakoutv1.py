"""
Breakout DQN Agent
==================
An implementation of a Deep Q-Network (DQN) agent to play Atari Breakout.

Features:
 - Experience replay buffer for diverse training samples
 - Separate target network to smooth out learning
 - Epsilon-greedy strategy for balanced exploration and exploitation
 - Lightweight CNN to extract actionable features from game frames
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ale_py

"""
Set random seeds for reproducibility
This ensures that every run (data shuffling, weight initialization, and action sampling)
follows the same "random" sequence, so experiments can be debuged and compared consistently
"""
RANDOM_SEED = 10
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


"""
Configuration parameters
key configurations:
    epsilon-greedy schedule: starts fully random (=1.0), decays to mostly greedy
    (=0.1) over 100k steps
    replay buffer: 50k transitions
    target network: synced every 10 episodes to stabilize learning
    train step: only once every 4 env steps
"""
CONFIG = {
    "env_id": "ALE/Breakout-v5",
    "frame_skip": 4,
    "frame_stack": 4,

    "total_episodes": 1000,
    "memory_size": 50000,
    "batch_size": 32,
    "target_update": 10,
    "train_frequency": 4,
    "gamma": 0.99,
    
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 100000,
    
    "learning_rate": 0.0001,
    
    "eval_frequency": 25,
    "eval_episodes": 3,
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "model_path": "../models/breakout_dqn.pt",
    "best_model_path": "../models/breakout_dqn_best.pt",
}

os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)

class ReplayMemory:
    """
    Experience replay memory to store transitions
    Stores (state, action, reward, next_state, done) tuples in a cyclic buffer. 
    On each training update, samples a random batch—breaking 
    temporal correlations and smoothing out learning.
    """
    def __init__(self, capacity=CONFIG["memory_size"]):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Deep Q-Network architecture for Atari
    convolutional layers:
     - 8x8 kernel, stride 4 -> 32 feature maps
     - 4x4 kernel, stride 2 -> 64 feature maps
     - 3x3 kernel, stride 1 -> 64 feature maps
    Fully connected layers:
     - Flattened output → 512 units → action-value outputs
     
    This is the part of the network that "looks" at raw Atari frames and extracts
    meaningful spacial features (walls, bricks, paddel, ball), before any decision
    is made.

    the first convolution takes the input image and slides a 8x8 window across it,
    moving 4 pixels at a time, applying 32 different filters of that size, producing
    32 feature maps...

    this exact sequence-8x8->4x4->3x3 with strides 4, 2, 1-was used in the DQN paper, 
    described on page 6 in the METHODS section under "model architecture"
    "It just works" - Tom Howard
    """
    def __init__(self, input_shape, num_actions):
        super().__init__()
        channels, height, width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Compute feature map size after convolutions
        dummy = torch.zeros(1, channels, height, width)
        conv_size = int(np.prod(self.features(dummy).size()))
        self.fc = nn.Sequential(
            nn.Linear(conv_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        # Remove any extra singleton dimensions
        if x.dim() == 5:
            x = x.squeeze(-1)
        elif x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BreakoutDQNAgent:
    """
    Agent managing interaction, replay buffer, and network updates.
    Two networks ensure stable target estimation.
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = CONFIG["device"]
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG["learning_rate"])
        self.memory = ReplayMemory()
        self.episode_count = 0
        self.step_count = 0
        self.best_eval_score = float('-inf')
    
    def select_action(self, state, evaluate=False):
        """
        Select an action using epsilon-greedy policy
        Evaluate mode: locks epsilon at 0.05 to add slight randomness
        during evaluation runs
        """
        # Calculate epsilon based on step count for more gradual decay
        epsilon = max(
            CONFIG["epsilon_end"],
            CONFIG["epsilon_start"] - 
            (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]) * 
            min(1.0, self.step_count / CONFIG["epsilon_decay"])
        )
        if evaluate:
            epsilon = 0.05

        if random.random() < epsilon:
            return random.randrange(self.policy_net.fc[-1].out_features)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
        return q_vals.max(1)[1].item()
    
    def update_target_network(self):
        """Copy policy network weights into target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def optimize_model(self):
        """
        Perform a single optimization step:
            1. Sample a batch from replay buffer
            2. Compute current Q-values for taken actions
            3. Compute target Q-values using target network
            4. Huber loss between current and target
            5. Backpropagate and clip gradients to [-1,1]
        """
        if len(self.memory) < CONFIG["batch_size"]:
            return None

        batch = self.memory.sample(CONFIG["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_current = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
        q_target = rewards + (1 - dones) * CONFIG["gamma"] * q_next

        loss = F.smooth_l1_loss(q_current.squeeze(), q_target)
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.policy_net.parameters():
            p.grad.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()
    
    def train(self, env):
        """
        Runs full training loop:
         - episodes of environment interaction
         - periodic optimization
         - target network sync every N episodes
         - evaluation and model checkpointing
        Returns episode rewards and loss history.
        """
        rewards_history, loss_history = [], []
        for ep in range(CONFIG["total_episodes"]):
            state, _ = env.reset(seed=random.randint(0, 1000))
            episode_reward, episode_losses = 0, []
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.memory.push(state, action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward
                self.step_count += 1
                if self.step_count % CONFIG["train_frequency"] == 0:
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_losses.append(loss)
            if (ep + 1) % CONFIG["target_update"] == 0:
                self.update_target_network()
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            rewards_history.append(episode_reward)
            loss_history.extend(episode_losses)
            print(f"Episode {ep+1}/{CONFIG['total_episodes']}  "
                  f"Reward {episode_reward:.1f}  "
                  f"Epsilon {max(CONFIG['epsilon_end'], CONFIG['epsilon_start'] - (CONFIG['epsilon_start']-CONFIG['epsilon_end'])*min(1, self.step_count/CONFIG['epsilon_decay'])):.2f}  "
                  f"AvgLoss {avg_loss:.4f}")
            if (ep + 1) % CONFIG["eval_frequency"] == 0:
                score = self.evaluate(env)
                if score > self.best_eval_score:
                    self.best_eval_score = score
                    torch.save(self.policy_net.state_dict(), CONFIG["best_model_path"])
                    print(f"New best model (score {score:.1f}) saved.")
        torch.save(self.policy_net.state_dict(), CONFIG["model_path"])
        print(f"Training complete. Models saved.")
        return rewards_history, loss_history

    def evaluate(self, env, num_episodes=CONFIG["eval_episodes"], render=False):
        """
        Runs a few episodes with near-greedy policy (ε=0.05).
        Prints and returns the average score.
        """
        total_score = 0
        for i in range(num_episodes):
            state, _ = env.reset(seed=i)
            episode_score = 0
            done = False
            while not done:
                if render:
                    env.render()
                action = self.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_score += reward
            total_score += episode_score
            print(f"Eval {i+1}/{num_episodes}: Score {episode_score:.1f}")
        avg = total_score / num_episodes
        print(f"Average eval score: {avg:.1f}")
        return avg

    def load(self, path):
        """Loads model weights from file and syncs target network."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")

    def save(self, path):
        """Saves policy network weights to file."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

def plot_training_results(rewards, losses, smoothing_window=10):
    """
    Generates and saves a plot of smoothed episode rewards and training loss.
    """
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    if len(rewards) >= smoothing_window:
        rewards_smoothed = np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        plt.plot(rewards_smoothed)
    else:
        plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 1, 2)
    if len(losses) >= smoothing_window:
        losses_smoothed = np.convolve(losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
        plt.plot(losses_smoothed)
    else:
        plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def create_breakout_env(render_mode=None):
    """
    Builds the Breakout environment with:
     - Atari preprocessing (no-ops, frame skip, resize, grayscale, normalize)
     - Frame stacking for temporal context
     - Automatic episode reward logging
    """
    env = gym.make(CONFIG["env_id"], render_mode=render_mode, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=CONFIG["frame_skip"],
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, CONFIG["frame_stack"])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def main():
    """
    Entry point: either loads a pretrained model for evaluation or trains from scratch.
    Prints environment specs and final evaluation results.
    """
    try:
        print(f"Env: {CONFIG['env_id']}")
        env = create_breakout_env()
        print(f"Obs space: {env.observation_space}, Action space: {env.action_space}")

        obs, _ = env.reset()
        print(f"Sample obs shape: {obs.shape}, dtype: {obs.dtype}")

        load_flag = os.path.exists(CONFIG["best_model_path"] ) and input("Load existing model? (y/n): ").lower() == 'y'
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n
        agent = BreakoutDQNAgent(state_dim, action_dim)

        if load_flag:
            agent.load(CONFIG["best_model_path"])
            render_env = create_breakout_env(render_mode="human")
            agent.evaluate(render_env, num_episodes=5, render=True)
            render_env.close()
        else:
            r, l = agent.train(env)
            plot_training_results(r, l)
            render_env = create_breakout_env(render_mode="human")
            agent.evaluate(render_env, num_episodes=5, render=True)
            render_env.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'env' in locals(): env.close()
        if 'render_env' in locals(): render_env.close()

if __name__ == "__main__":
    from PIL import Image
    main()