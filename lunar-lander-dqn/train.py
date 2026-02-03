"""
LunarLander-v3 DQN 训练脚本
目标: 平均 reward > 200
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import imageio
import time
import os

from dqn import DQNAgent, device

# 配置
MAX_EPISODES = 1000
TARGET_REWARD = 200
WINDOW_SIZE = 100  # 计算平均奖励的窗口
SAVE_INTERVAL = 100

def train():
    print(f"Using device: {device}")
    
    # 创建环境
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # 创建 Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=10
    )
    
    # 记录
    rewards_history = []
    avg_rewards_history = []
    recent_rewards = deque(maxlen=WINDOW_SIZE)
    best_avg_reward = -float('inf')
    
    start_time = time.time()
    
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            episode_reward += reward
        
        agent.decay_epsilon()
        
        rewards_history.append(episode_reward)
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        avg_rewards_history.append(avg_reward)
        
        # 打印进度
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Avg(100): {avg_reward:7.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")
        
        # 保存最佳模型
        if avg_reward > best_avg_reward and len(recent_rewards) == WINDOW_SIZE:
            best_avg_reward = avg_reward
            agent.save("model_best.pth")
        
        # 定期保存
        if episode % SAVE_INTERVAL == 0:
            agent.save(f"model_ep{episode}.pth")
        
        # 检查是否达标
        if avg_reward >= TARGET_REWARD and len(recent_rewards) == WINDOW_SIZE:
            print(f"\n🎉 Solved in {episode} episodes! Avg reward: {avg_reward:.1f}")
            agent.save("model.pth")
            break
    
    # 保存最终模型
    agent.save("model.pth")
    env.close()
    
    # 绘制训练曲线
    plot_training_curve(rewards_history, avg_rewards_history)
    
    return agent, rewards_history

def plot_training_curve(rewards, avg_rewards):
    """绘制并保存训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    plt.plot(avg_rewards, color='red', linewidth=2, label=f'Avg (window={WINDOW_SIZE})')
    plt.axhline(y=TARGET_REWARD, color='green', linestyle='--', label=f'Target ({TARGET_REWARD})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('LunarLander-v3 DQN Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最后100轮的分布
    plt.subplot(1, 2, 2)
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    plt.hist(last_100, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(last_100), color='red', linestyle='--', 
                label=f'Mean: {np.mean(last_100):.1f}')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('Last 100 Episodes Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    plt.close()
    print("✅ Saved training_curve.png")

def record_gif(model_path="model.pth", output_path="lunar_lander_trained.gif", episodes=3):
    """录制训练好的模型"""
    print(f"\n📹 Recording GIF with {model_path}...")
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    
    frames = []
    total_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            frames.append(env.render())
            action = agent.select_action(obs, greedy=True)
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            
            if done or trunc:
                # 添加几帧结束画面
                for _ in range(15):
                    frames.append(env.render())
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.1f}")
    
    env.close()
    
    # 保存 GIF
    imageio.mimsave(output_path, frames, fps=30)
    print(f"✅ Saved {output_path} ({len(frames)} frames)")
    print(f"   Average reward: {np.mean(total_rewards):.1f}")
    
    return total_rewards

if __name__ == "__main__":
    # 切换到脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("  LunarLander-v3 DQN Training")
    print("=" * 60)
    
    # 训练
    agent, rewards = train()
    
    # 录制 GIF
    record_gif()
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
