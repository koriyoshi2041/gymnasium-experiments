#!/usr/bin/env python3
"""
BipedalWalker-v3 PPO Training Script
使用 Stable-Baselines3 训练双足机器人走路

作者: Rios
日期: 2025-07
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 训练参数
TOTAL_TIMESTEPS = 1_000_000  # 1M steps
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
MODEL_SAVE_PATH = "bipedal_ppo_model"
LOG_DIR = "./logs/"

os.makedirs(LOG_DIR, exist_ok=True)


class TrainingCallback(BaseCallback):
    """自定义回调：记录训练曲线"""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.timesteps = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # 累积当前 episode 奖励
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            
        # 检查 episode 是否结束
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            # 定期记录
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                self.rewards.append(avg_reward)
                self.timesteps.append(self.num_timesteps)
                
                if self.verbose:
                    print(f"Timesteps: {self.num_timesteps:,} | Episodes: {len(self.episode_rewards)} | Avg Reward (last 100): {avg_reward:.2f}")
        
        return True
    
    def get_training_data(self):
        return self.timesteps, self.rewards


def create_env():
    """创建训练环境"""
    env = gym.make("BipedalWalker-v3")
    env = Monitor(env, LOG_DIR)
    return env


def train():
    """主训练函数"""
    print("=" * 60)
    print("🦿 BipedalWalker-v3 PPO Training")
    print("=" * 60)
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    
    # 创建向量化环境
    env = make_vec_env("BipedalWalker-v3", n_envs=4, seed=SEED)
    
    # 创建评估环境
    eval_env = make_vec_env("BipedalWalker-v3", n_envs=1, seed=SEED + 100)
    
    # PPO 超参数（针对 BipedalWalker 优化）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,  # 禁用 tensorboard
        seed=SEED,
    )
    
    # 回调
    training_callback = TrainingCallback(check_freq=1000, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    
    # 开始训练
    print("\n🏃 Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[training_callback, eval_callback],
        progress_bar=True,
    )
    
    # 保存模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}.zip")
    
    # 绘制训练曲线
    plot_training_curve(training_callback)
    
    # 录制 GIF
    record_gif(model)
    
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)


def plot_training_curve(callback):
    """绘制训练曲线"""
    timesteps, rewards = callback.get_training_data()
    
    if len(timesteps) < 2:
        print("⚠️ Not enough data to plot training curve")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, rewards, 'b-', alpha=0.7, label='Avg Reward (100 eps)')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title('BipedalWalker-v3 PPO Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 平滑曲线
    plt.subplot(1, 2, 2)
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(timesteps[:len(smoothed)], smoothed, 'r-', linewidth=2, label='Smoothed (10-pt avg)')
    plt.plot(timesteps, rewards, 'b-', alpha=0.3, label='Raw')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Smoothed Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print("📊 Training curve saved to training_curve.png")
    plt.close()


def record_gif(model, filename="bipedal_walker.gif", n_frames=500):
    """录制 GIF 动画"""
    print(f"\n🎬 Recording GIF ({n_frames} frames)...")
    
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    frames = []
    
    obs, _ = env.reset(seed=SEED)
    total_reward = 0
    episode_count = 0
    
    for i in range(n_frames):
        frame = env.render()
        frames.append(frame)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            episode_count += 1
            print(f"  Episode {episode_count} reward: {total_reward:.2f}")
            total_reward = 0
            obs, _ = env.reset()
    
    env.close()
    
    # 保存 GIF
    imageio.mimsave(filename, frames, fps=30, loop=0)
    print(f"✅ GIF saved to {filename}")
    print(f"   - Frames: {len(frames)}")
    print(f"   - Duration: {len(frames)/30:.1f}s")


def evaluate_model(model_path=MODEL_SAVE_PATH):
    """评估已训练的模型"""
    print("\n📊 Evaluating model...")
    
    model = PPO.load(model_path)
    env = gym.make("BipedalWalker-v3")
    
    rewards = []
    for ep in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.2f}")
    
    env.close()
    
    print(f"\n📈 Results over 10 episodes:")
    print(f"   Mean: {np.mean(rewards):.2f}")
    print(f"   Std:  {np.std(rewards):.2f}")
    print(f"   Min:  {np.min(rewards):.2f}")
    print(f"   Max:  {np.max(rewards):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BipedalWalker PPO Training")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--gif", action="store_true", help="Record GIF only")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total training timesteps")
    
    args = parser.parse_args()
    
    if args.timesteps != TOTAL_TIMESTEPS:
        TOTAL_TIMESTEPS = args.timesteps
    
    if args.gif:
        model = PPO.load(MODEL_SAVE_PATH)
        record_gif(model)
    elif args.eval:
        evaluate_model()
    else:
        train()
