"""录制训练好的 Agent"""
import gymnasium as gym
import imageio
import numpy as np
from dqn import DQNAgent

def record_gif(model_path="model.pth", output_path="lunar_lander_trained.gif", episodes=3):
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
    record_gif("model.pth", "lunar_lander_trained.gif", episodes=3)
