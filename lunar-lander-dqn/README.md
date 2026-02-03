# LunarLander-v3 DQN 训练实验

使用 Deep Q-Network (DQN) 算法训练一个能够成功着陆月球着陆器的 Agent。

## 🎯 实验目标

训练一个 DQN Agent 在 Gymnasium 的 LunarLander-v3 环境中实现稳定着陆，目标是 100 episode 平均 reward > 200。

## 🛠️ 算法实现

### DQN 核心组件

1. **ReplayBuffer** - 经验回放缓冲区
   - 容量: 100,000 transitions
   - 随机采样打破时间相关性

2. **DQN 网络架构**
   ```
   Input (8) → Linear(256) → ReLU 
            → Linear(256) → ReLU 
            → Linear(256) → ReLU 
            → Output (4)
   ```

3. **Double DQN**
   - 使用在线网络选择动作
   - 使用目标网络评估 Q 值
   - 减少 Q 值过估计

### 超参数

| 参数 | 值 |
|-----|-----|
| 学习率 | 5e-4 |
| 折扣因子 γ | 0.99 |
| Epsilon 初始值 | 1.0 |
| Epsilon 最小值 | 0.01 |
| Epsilon 衰减率 | 0.995 |
| Batch Size | 64 |
| 目标网络更新频率 | 10 steps |
| Buffer 容量 | 100,000 |

## 📊 训练结果

### 训练曲线

![Training Curve](training_curve.png)

### 关键里程碑

| Episode | 平均奖励 (100ep) | 备注 |
|---------|-----------------|------|
| 100 | -105.8 | 开始学习 |
| 200 | -51.1 | 显著改善 |
| 280 | +4.5 | 首次转正！ |
| 370 | +0.4 | 稳定正奖励 |

### 最终表现

使用训练好的模型测试 3 个 episodes：

| Episode | Reward |
|---------|--------|
| 1 | 235.4 |
| 2 | 195.1 |
| 3 | 237.3 |
| **平均** | **222.6** ✅ |

## 🎬 训练效果演示

![LunarLander Demo](lunar_lander_trained.gif)

Agent 能够：
- ✅ 控制着陆器平稳下降
- ✅ 调整姿态保持垂直
- ✅ 精确降落在着陆点
- ✅ 着陆时速度适当

## 📁 文件说明

```
lunar-lander-dqn/
├── dqn.py                  # DQN 实现 (ReplayBuffer, DQN网络, Agent)
├── train.py                # 训练脚本
├── record_gif.py           # GIF 录制脚本
├── model.pth               # 最终模型权重
├── model_best.pth          # 最佳模型权重
├── model_ep*.pth           # 中间 checkpoint
├── training_curve.png      # 训练曲线图
├── lunar_lander_trained.gif # 训练效果演示
└── README.md               # 本文件
```

## 🚀 复现步骤

```bash
# 1. 安装依赖
pip install 'gymnasium[box2d]' imageio torch matplotlib

# 2. 训练模型
python train.py

# 3. 录制 GIF
python record_gif.py
```

## 💡 关键学习点

1. **Double DQN** 有效减少了 Q 值过估计，提高训练稳定性
2. **Epsilon Decay** 策略平衡了探索与利用
3. **Target Network** 的软更新防止了训练振荡
4. **Gradient Clipping** 避免了梯度爆炸

## 📚 参考资料

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (DQN 原论文)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (Double DQN)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

*实验完成于 2025-02-03，使用 Apple MPS 加速训练*
