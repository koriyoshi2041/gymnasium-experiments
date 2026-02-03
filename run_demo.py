#!/usr/bin/env python3
"""
Gymnasium RL Experiments - Demo Runner
一键运行三个实验的演示
"""
import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

def run_lunar_lander():
    """运行 LunarLander DQN 演示"""
    print("🚀 LunarLander-v3 + DQN")
    
    class DQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 4)
            )
        def forward(self, x):
            return self.net(x)
    
    model = DQN()
    checkpoint = torch.load('lunar-lander-dqn/model_best.pth', weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['q_network'])
    model.eval()
    
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        with torch.no_grad():
            q = model(torch.FloatTensor(obs).unsqueeze(0))
            action = q.argmax(dim=1).item()
        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        if done or trunc:
            print(f"Episode finished! Reward: {total_reward:.1f}")
            break
    env.close()

def run_cartpole():
    """运行 CartPole REINFORCE 演示"""
    print("🎯 CartPole-v1 + REINFORCE")
    
    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(4, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 2), nn.Softmax(dim=-1)
            )
        def forward(self, x):
            return self.network(x)
    
    model = PolicyNetwork()
    model.load_state_dict(torch.load('cartpole-reinforce/policy_model.pth', weights_only=True, map_location='cpu'))
    model.eval()
    
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        with torch.no_grad():
            probs = model(torch.FloatTensor(obs).unsqueeze(0))
            action = probs.argmax(dim=1).item()
        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        if done or trunc:
            print(f"Episode finished! Steps: {int(total_reward)}")
            break
    env.close()

def run_bipedal():
    """运行 BipedalWalker PPO 演示"""
    print("🦿 BipedalWalker-v3 + PPO")
    
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("请安装 stable-baselines3: pip install stable-baselines3")
        return
    
    model = PPO.load("bipedal-ppo/bipedal_ppo_model.zip")
    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        if done or trunc:
            print(f"Episode finished! Reward: {total_reward:.1f}")
            break
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Run RL experiment demos')
    parser.add_argument('--env', type=str, choices=['lunar', 'cartpole', 'bipedal', 'all'],
                        default='all', help='Which experiment to run')
    args = parser.parse_args()
    
    if args.env == 'lunar' or args.env == 'all':
        run_lunar_lander()
    if args.env == 'cartpole' or args.env == 'all':
        run_cartpole()
    if args.env == 'bipedal' or args.env == 'all':
        run_bipedal()

if __name__ == '__main__':
    main()
