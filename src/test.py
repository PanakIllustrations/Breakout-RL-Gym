import os
import time
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import ale_py

RANDOM_SEED = 10
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

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
    "video_dir": "../videos/breakout",
}

os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
os.makedirs(CONFIG["video_dir"], exist_ok=True)
class DQN(nn.Module):
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
        dummy = torch.zeros(1, channels, height, width)
        conv_size = int(np.prod(self.features(dummy).size()))
        self.fc = nn.Sequential(
            nn.Linear(conv_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(-1)
        elif x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BreakoutAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = CONFIG["device"]
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        
        print(f"Loaded model from {model_path}")
    
    def select_action(self, state, evaluate=True):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            if evaluate and random.random() < 0.05:
                return random.randrange(self.action_dim)
            values = q_values.cpu().numpy()[0]
            return q_values.max(1)[1].item()

def create_breakout_env(render_mode=None):
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
    
    return env

def create_recording_env():
    env = gym.make(CONFIG["env_id"], render_mode="rgb_array", frameskip=1)
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_path = os.path.join(CONFIG["video_dir"], f"breakout-{timestamp}")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_path,
        episode_trigger=lambda episode_id: True,
        name_prefix="breakout"
    )
    print(f"Recording videos to {video_path}")
    return env

def test_agent(model_path=None, episodes=5, render_mode="human", record=True):
    if model_path is None:
        if os.path.exists(CONFIG["best_model_path"]):
            model_path = CONFIG["best_model_path"]
            print(f"Using best model: {model_path}")
        elif os.path.exists(CONFIG["model_path"]):
            model_path = CONFIG["model_path"]
            print(f"Using standard model: {model_path}")
        else:
            print("No trained model found!")
            return
    
    display_env = create_breakout_env(render_mode=render_mode)
    recording_env = create_recording_env() if record else None
    
    print(f"Observation space: {display_env.observation_space}")
    print(f"Action space: {display_env.action_space}")
    
    state_dim = display_env.observation_space.shape
    action_dim = display_env.action_space.n
    print(f"State shape: {state_dim}")
    
    agent = BreakoutAgent(state_dim, action_dim, model_path)
    
    total_reward = 0
    action_names = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}
    
    for episode in range(episodes):
        seed = episode + 42
        display_state, _ = display_env.reset(seed=seed)
        
        if recording_env:
            recording_state, _ = recording_env.reset(seed=seed)
        
        episode_reward = 0
        done_display = False
        done_recording = False
        step = 0
        
        action_counts = [0] * action_dim

        while not done_display:
            step += 1
            
            action = agent.select_action(display_state)
            action_counts[action] += 1

            next_display_state, reward, terminated, truncated, _ = display_env.step(action)
            done_display = terminated or truncated
            
            if recording_env and not done_recording:
                next_recording_state, _, rec_terminated, rec_truncated, _ = recording_env.step(action)
                done_recording = rec_terminated or rec_truncated
                recording_state = next_recording_state
            
            display_state = next_display_state
            episode_reward += reward
            
            if step % 30 == 0 and render_mode == "human":
                print(f"Step {step}, Action: {action_names.get(action, action)}")
        
        action_dist = [count/max(1, sum(action_counts)) for count in action_counts]
        action_dist_str = " ".join([f"{action_names.get(i, i)}:{dist:.2f}" for i, dist in enumerate(action_dist)])
        
        total_reward += episode_reward
        print(f"Episode {episode+1}/{episodes} - Reward: {episode_reward:.2f}, Steps: {step}")
        print(f"Actions: {action_dist_str}")
    
    avg_reward = total_reward / episodes
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")
    
    display_env.close()
    if recording_env:
        recording_env.close()
    
    return avg_reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a trained Breakout DQN agent")
    parser.add_argument("--model", type=str, default=None, help="Path to model file (default: best_model)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--no-record", action="store_true", help="Disable video recording")
    
    args = parser.parse_args()
    
    render_mode = None if args.no_render else "human"
    record = not args.no_record
    
    test_agent(
        model_path=args.model,
        episodes=args.episodes,
        render_mode=render_mode,
        record=record
    )