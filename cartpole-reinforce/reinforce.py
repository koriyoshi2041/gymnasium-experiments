#!/usr/bin/env python3
"""
REINFORCE (Policy Gradient) 算法实现 - CartPole 环境

REINFORCE 是最基础的策略梯度算法：
1. 用神经网络直接输出动作概率（策略 π(a|s)）
2. 采样完整轨迹后，用蒙特卡洛回报更新策略
3. 目标：最大化期望累积回报 E[Σγ^t * r_t]

核心公式：
    ∇J(θ) ≈ Σ_t [∇log π(a_t|s_t; θ) * G_t]
    其中 G_t = Σ_{k=t}^T γ^(k-t) * r_k 是从 t 时刻开始的折扣回报
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import imageio
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 策略网络 (Policy Network)
# ============================================================
class PolicyNetwork(nn.Module):
    """
    策略网络：输入状态 s，输出动作概率分布 π(·|s)
    
    CartPole 状态空间：4维（位置、速度、角度、角速度）
    CartPole 动作空间：2个离散动作（左推、右推）
    
    网络结构：简单的两层全连接网络
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 128, action_dim: int = 2):
        super(PolicyNetwork, self).__init__()
        
        # 两层全连接 + ReLU 激活
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 输入层 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 隐藏层 -> 输出层
            nn.Softmax(dim=-1)                  # Softmax 输出概率分布
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播：状态 -> 动作概率"""
        return self.network(state)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        根据当前状态选择动作
        
        Args:
            state: 环境状态（numpy数组）
            deterministic: 是否确定性选择（选概率最大的动作）
        
        Returns:
            选择的动作
        """
        # 转换为 tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率
        with torch.no_grad():
            action_probs = self.forward(state_tensor)
        
        if deterministic:
            # 确定性：选概率最大的
            action = torch.argmax(action_probs, dim=1).item()
        else:
            # 随机性：按概率采样
            dist = Categorical(action_probs)
            action = dist.sample().item()
        
        return action


# ============================================================
# REINFORCE Agent
# ============================================================
class REINFORCEAgent:
    """
    REINFORCE 算法实现
    
    算法流程：
    1. 收集一整条轨迹 (s_0, a_0, r_1, s_1, a_1, r_2, ...)
    2. 计算每个时刻的折扣回报 G_t
    3. 计算策略梯度并更新参数
    
    改进：使用 baseline（回报标准化）减少方差
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, 
                 hidden_dim: int = 128, learning_rate: float = 1e-3, 
                 gamma: float = 0.99):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层大小
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.gamma = gamma
        
        # 初始化策略网络
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        
        # Adam 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 存储轨迹数据
        self.saved_log_probs = []  # 动作的 log 概率
        self.rewards = []           # 获得的奖励
    
    def select_action(self, state: np.ndarray) -> int:
        """
        选择动作并保存 log 概率（用于后续梯度计算）
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        
        # 创建分类分布
        dist = Categorical(action_probs)
        
        # 采样动作
        action = dist.sample()
        
        # 保存 log 概率（梯度会通过这里反向传播）
        self.saved_log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward: float):
        """存储奖励"""
        self.rewards.append(reward)
    
    def compute_returns(self) -> torch.Tensor:
        """
        计算折扣回报 G_t = Σ_{k=t}^T γ^(k-t) * r_k
        
        从后往前计算，效率更高：
        G_T = r_T
        G_{T-1} = r_{T-1} + γ * G_T
        G_{T-2} = r_{T-2} + γ * G_{T-1}
        ...
        """
        returns = []
        G = 0
        
        # 从最后一个时刻往前算
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 标准化（减少方差，这是一种简单的 baseline）
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self) -> float:
        """
        更新策略网络
        
        策略梯度：∇J(θ) ≈ Σ_t [∇log π(a_t|s_t; θ) * G_t]
        
        PyTorch 中：
        - 我们要最大化 J(θ)，等价于最小化 -J(θ)
        - loss = -Σ_t [log π(a_t|s_t) * G_t]
        """
        # 计算折扣回报
        returns = self.compute_returns()
        
        # 计算 policy loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            # 负号：因为我们要最大化期望回报，但优化器做的是最小化
            policy_loss.append(-log_prob * G)
        
        # 合并所有时刻的 loss
        loss = torch.stack(policy_loss).sum()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 清空轨迹缓存
        loss_value = loss.item()
        self.saved_log_probs = []
        self.rewards = []
        
        return loss_value


# ============================================================
# 训练函数
# ============================================================
def train(learning_rate: float = 1e-3, 
          num_episodes: int = 2000,
          target_reward: float = 495.0,
          patience: int = 100,
          verbose: bool = True) -> tuple:
    """
    训练 REINFORCE agent
    
    Args:
        learning_rate: 学习率
        num_episodes: 最大训练回合数
        target_reward: 目标分数（CartPole 满分 500）
        patience: 连续达到目标分数的次数才算"稳定"
        verbose: 是否打印训练过程
    
    Returns:
        (agent, rewards_history)
    """
    # 创建环境
    env = gym.make("CartPole-v1")
    
    # 创建 agent
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=learning_rate
    )
    
    # 记录训练历史
    rewards_history = []
    recent_rewards = deque(maxlen=patience)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        # 收集一整条轨迹
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储奖励
            agent.store_reward(reward)
            episode_reward += reward
            
            state = next_state
            
            if done:
                break
        
        # 更新策略
        agent.update()
        
        # 记录
        rewards_history.append(episode_reward)
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        
        # 打印进度
        if verbose and (episode + 1) % 100 == 0:
            print(f"[lr={learning_rate}] Episode {episode+1}, "
                  f"Avg Reward (last {patience}): {avg_reward:.1f}")
        
        # 检查是否稳定达到目标
        if len(recent_rewards) >= patience and avg_reward >= target_reward:
            if verbose:
                print(f"[lr={learning_rate}] 🎉 Solved in {episode+1} episodes! "
                      f"Avg: {avg_reward:.1f}")
            break
    
    env.close()
    return agent, rewards_history


# ============================================================
# 评估和录制 GIF
# ============================================================
def record_gif(policy: PolicyNetwork, filename: str = "cartpole_reinforce.gif"):
    """
    用训练好的策略录制 GIF
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    
    obs, _ = env.reset()
    
    for _ in range(500):
        frames.append(env.render())
        action = policy.act(obs, deterministic=True)  # 使用确定性策略
        obs, r, done, trunc, _ = env.step(action)
        if done or trunc:
            break
    
    env.close()
    
    # 保存 GIF
    imageio.mimsave(filename, frames, fps=30)
    print(f"✅ GIF saved: {filename} ({len(frames)} frames)")
    
    return len(frames)


# ============================================================
# 绘制训练曲线
# ============================================================
def plot_training_curves(results: dict, filename: str = "training_curve.png"):
    """
    绘制不同学习率的训练曲线对比图
    
    Args:
        results: {learning_rate: rewards_history}
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # 绿、蓝、红
    
    for i, (lr, rewards) in enumerate(results.items()):
        episodes = range(1, len(rewards) + 1)
        
        # 原始曲线（透明度低）
        plt.plot(episodes, rewards, alpha=0.2, color=colors[i])
        
        # 平滑曲线（移动平均）
        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window, len(rewards)+1), smoothed, 
                    label=f'lr={lr}', color=colors[i], linewidth=2)
        else:
            plt.plot(episodes, rewards, label=f'lr={lr}', color=colors[i])
    
    plt.axhline(y=500, color='gold', linestyle='--', label='Target (500)', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('REINFORCE on CartPole-v1: Learning Rate Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Training curve saved: {filename}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("REINFORCE (Policy Gradient) - CartPole Experiment")
    print("=" * 60)
    
    # 要对比的学习率
    learning_rates = [1e-2, 3e-3, 1e-3]
    
    # 存储结果
    results = {}
    best_agent = None
    best_lr = None
    best_episodes = float('inf')
    
    # 训练不同学习率
    for lr in learning_rates:
        print(f"\n{'='*40}")
        print(f"Training with learning rate = {lr}")
        print(f"{'='*40}")
        
        agent, rewards = train(
            learning_rate=lr,
            num_episodes=2000,
            target_reward=495.0,
            patience=100,
            verbose=True
        )
        
        results[lr] = rewards
        
        # 记录最好的模型（最快达到目标的）
        if len(rewards) < best_episodes:
            best_episodes = len(rewards)
            best_agent = agent
            best_lr = lr
    
    print(f"\n{'='*60}")
    print(f"🏆 Best learning rate: {best_lr} (solved in {best_episodes} episodes)")
    print(f"{'='*60}")
    
    # 绘制训练曲线对比图
    plot_training_curves(results, "training_curve.png")
    
    # 保存最好的模型
    torch.save(best_agent.policy.state_dict(), "policy_model.pth")
    print(f"✅ Best model saved: policy_model.pth")
    
    # 用最好的模型录制 GIF
    print(f"\n📹 Recording GIF with best model (lr={best_lr})...")
    record_gif(best_agent.policy, "cartpole_reinforce.gif")
    
    print("\n" + "=" * 60)
    print("✅ All outputs generated:")
    print("   - reinforce.py")
    print("   - policy_model.pth")
    print("   - training_curve.png")
    print("   - cartpole_reinforce.gif")
    print("=" * 60)


if __name__ == "__main__":
    main()
