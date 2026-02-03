# REINFORCE (Policy Gradient) - CartPole 实验

## 项目概述

本项目从零实现 REINFORCE 算法，并在 CartPole-v1 环境上训练直到稳定达到满分（500分）。

## 什么是 REINFORCE？

REINFORCE 是最基础的**策略梯度（Policy Gradient）**算法，由 Ronald Williams 在 1992 年提出。

### 核心思想

与 DQN 等 **值函数方法** 不同，策略梯度方法**直接学习策略**：

```
值函数方法:  状态 s → Q(s,a) → 选择 max Q 的动作
策略梯度:    状态 s → π(a|s) → 按概率直接采样动作
```

### 数学推导

**目标：最大化期望累积回报**

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

其中 $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, ...)$ 是一条轨迹。

**策略梯度定理：**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 $t$ 开始的折扣回报。

### 算法流程

```
1. 用当前策略 π_θ 采样一条完整轨迹
2. 对轨迹中每个时刻 t：
   - 计算折扣回报 G_t
   - 计算梯度 ∇log π(a_t|s_t) * G_t
3. 更新参数 θ ← θ + α * Σ梯度
4. 重复直到收敛
```

## 代码结构

```
cartpole-reinforce/
├── reinforce.py         # REINFORCE 完整实现
├── policy_model.pth     # 训练好的模型权重
├── training_curve.png   # 训练曲线（含 lr 对比）
├── cartpole_reinforce.gif  # 效果展示 GIF
└── README.md            # 本文档
```

## 关键实现细节

### 1. 策略网络

```python
class PolicyNetwork(nn.Module):
    """
    输入: 状态 (4维: 位置、速度、角度、角速度)
    输出: 动作概率分布 (2维: 左推、右推)
    """
    def __init__(self):
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)  # 输出概率
        )
```

### 2. 折扣回报计算

```python
def compute_returns(rewards, gamma=0.99):
    """从后往前计算更高效"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
```

### 3. 回报标准化（Baseline）

为了减少方差，我们对回报进行标准化：

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

这是一种简单的 baseline，可以显著加速收敛。

### 4. 策略梯度损失

```python
# 最大化期望回报 = 最小化负期望回报
loss = -Σ [log π(a_t|s_t) * G_t]
```

## 学习率对比实验

测试了三个学习率：`1e-2`, `3e-3`, `1e-3`

| 学习率 | 收敛回合数 | 特点 |
|--------|-----------|------|
| 1e-2   | ~300-500  | 收敛快，但可能不稳定 |
| 3e-3   | ~400-700  | 平衡：速度适中，稳定性好 |
| 1e-3   | ~600-1000 | 收敛慢，但通常更稳定 |

实际结果可能因随机种子而异。

## CartPole 环境说明

- **状态空间：** 4维连续 (位置, 速度, 角度, 角速度)
- **动作空间：** 2个离散动作 (左推=0, 右推=1)
- **奖励：** 每存活一步 +1
- **终止条件：** 杆子倾斜超过 ±12°，或小车移出边界
- **满分：** 500 (坚持 500 步)

## 运行方法

```bash
# 安装依赖
pip install gymnasium torch matplotlib imageio

# 运行训练
python reinforce.py
```

## REINFORCE 的优缺点

### 优点 ✅
- 实现简单，概念清晰
- 可处理连续动作空间
- 直接优化目标函数

### 缺点 ❌
- **高方差：** 使用完整轨迹的蒙特卡洛估计
- **样本效率低：** 每条轨迹只能用一次
- **只能在线学习：** 不能使用经验回放

### 改进方向
- **Actor-Critic：** 用价值网络估计 baseline
- **A2C/A3C：** 并行采样，减少方差
- **PPO/TRPO：** 限制更新步长，提高稳定性

## 参考资料

- [Policy Gradient Methods for RL (Sutton et al., 1999)](http://incompleteideas.net/papers/SMSM99.pdf)
- [Simple Statistical Gradient-Following Algorithms (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [OpenAI Spinning Up - Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

---

*Generated with REINFORCE implementation experiment*
