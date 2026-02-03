#!/usr/bin/env python3
"""一键运行三个 Gymnasium 实验的演示"""

import argparse
import os
import sys

def run_lunar_lander():
    """运行 Lunar Lander DQN 演示"""
    import gymnasium as gym
    import torch
    import torch.nn as nn
    
    class DQN(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    env = gym.make("LunarLander-v3", render_mode="human")
    model = DQN(8, 4)
    model_path = os.path.join(os.path.dirname(__file__), "lunar-lander-dqn/model_best.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action = model(torch.FloatTensor(state)).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"🚀 Lunar Lander 总奖励: {total_reward:.2f}")
    env.close()

def run_cartpole():
    """运行 CartPole REINFORCE 演示"""
    import gymnasium as gym
    import torch
    import torch.nn as nn
    
    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=-1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    env = gym.make("CartPole-v1", render_mode="human")
    model = PolicyNetwork()
    model_path = os.path.join(os.path.dirname(__file__), "cartpole-reinforce/policy_model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            probs = model(torch.FloatTensor(state))
            action = torch.multinomial(probs, 1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"🎯 CartPole 总奖励: {total_reward:.0f}")
    env.close()

def run_bipedal():
    """运行 BipedalWalker PPO 演示"""
    import gymnasium as gym
    from stable_baselines3 import PPO
    
    env = gym.make("BipedalWalker-v3", render_mode="human")
    model_path = os.path.join(os.path.dirname(__file__), "bipedal-ppo/bipedal_ppo_model.zip")
    model = PPO.load(model_path)
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"🚶 BipedalWalker 总奖励: {total_reward:.2f}")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="运行 Gymnasium RL 实验演示")
    parser.add_argument("--env", choices=["lunar", "cartpole", "bipedal", "all"], 
                        default="all", help="选择要运行的环境")
    args = parser.parse_args()
    
    runners = {
        "lunar": ("🚀 Lunar Lander (DQN)", run_lunar_lander),
        "cartpole": ("🎯 CartPole (REINFORCE)", run_cartpole),
        "bipedal": ("🚶 BipedalWalker (PPO)", run_bipedal),
    }
    
    if args.env == "all":
        for name, (desc, runner) in runners.items():
            print(f"\n{'='*50}")
            print(f"运行 {desc}")
            print('='*50)
            runner()
    else:
        desc, runner = runners[args.env]
        print(f"运行 {desc}")
        runner()

if __name__ == "__main__":
    main()
