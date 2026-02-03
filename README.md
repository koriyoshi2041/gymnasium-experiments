# Gymnasium RL Experiments

三个经典强化学习实验，覆盖 Value-based、Policy-based 和 Actor-Critic 方法。

## 🎮 实验列表

| 实验 | 环境 | 算法 | 核心技术 |
|------|------|------|----------|
| 🚀 LunarLander | 月球着陆器 | DQN | 经验回放, Target Network |
| 🎯 CartPole | 平衡杆 | REINFORCE | Policy Gradient |
| 🦿 BipedalWalker | 双足行走 | PPO | Clipped Objective, GAE |

## 📦 安装

```bash
pip install gymnasium[box2d] torch stable-baselines3 imageio matplotlib
```

## 🚀 快速开始

```bash
# 运行所有演示
python run_demo.py --env all

# 运行单个实验
python run_demo.py --env lunar      # 月球着陆
python run_demo.py --env cartpole   # 平衡杆
python run_demo.py --env bipedal    # 双足行走
```

## 📁 项目结构

```
gymnasium-experiments/
├── run_demo.py                    # 一键演示脚本
├── lunar-lander-dqn/
│   ├── dqn.py                     # DQN 实现
│   ├── train.py                   # 训练脚本
│   ├── model_best.pth             # 训练好的模型
│   ├── training_curve.png         # 训练曲线
│   └── lunar_lander_trained.gif   # 演示 GIF
├── cartpole-reinforce/
│   ├── reinforce.py               # REINFORCE 实现
│   ├── policy_model.pth           # 训练好的模型
│   ├── training_curve.png         # LR 对比曲线
│   └── cartpole_reinforce.gif     # 演示 GIF
└── bipedal-ppo/
    ├── train.py                   # PPO 训练脚本
    ├── bipedal_ppo_model.zip      # SB3 模型
    └── bipedal_walker.gif         # 演示 GIF
```

## 🎬 演示效果

### LunarLander (DQN)
控制着陆器在月球表面安全降落。目标 reward > 200。

### CartPole (REINFORCE)
通过左右移动小车保持杆子平衡 500 步。

### BipedalWalker (PPO)
控制双足机器人行走。目标 reward > 300。

## 📊 训练结果

- **LunarLander**: ~200 episodes 达到 reward > 200
- **CartPole**: ~700 episodes 达到满分 500
- **BipedalWalker**: ~800K steps 达到 reward > 250

## License

MIT
