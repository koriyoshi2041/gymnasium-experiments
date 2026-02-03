# Gymnasium RL Experiments 🎮

三个经典强化学习实验，使用不同算法解决 Gymnasium 环境。

## 实验概览

| 环境 | 算法 | 描述 |
|------|------|------|
| 🚀 Lunar Lander | DQN | 深度 Q 网络，控制着陆器安全着陆 |
| 🎯 CartPole | REINFORCE | 策略梯度，平衡倒立摆 |
| 🚶 BipedalWalker | PPO | 近端策略优化，双足机器人行走 |

## 安装依赖

```bash
pip install gymnasium[box2d] torch stable-baselines3 imageio
```

## 快速开始

```bash
# 运行所有演示
python run_demo.py

# 运行单个环境
python run_demo.py --env lunar
python run_demo.py --env cartpole
python run_demo.py --env bipedal
```

## 目录结构

```
gymnasium-experiments/
├── run_demo.py              # 一键演示脚本
├── lunar-lander-dqn/        # DQN 实验
│   ├── dqn.py               # DQN 网络定义
│   ├── train.py             # 训练脚本
│   └── model_best.pth       # 最佳模型
├── cartpole-reinforce/      # REINFORCE 实验
│   ├── reinforce.py         # 策略网络和训练
│   └── policy_model.pth     # 训练好的策略
└── bipedal-ppo/             # PPO 实验
    ├── train.py             # 训练脚本
    └── bipedal_ppo_model.zip # SB3 模型
```

## 训练结果

### Lunar Lander (DQN)
![训练曲线](lunar-lander-dqn/training_curve.png)

### CartPole (REINFORCE)
![训练曲线](cartpole-reinforce/training_curve.png)

### BipedalWalker (PPO)
![训练曲线](bipedal-ppo/training_curve.png)

## License

MIT
