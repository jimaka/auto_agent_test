<!-- [![管道状态](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/pipeline.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) -->
![CI](https://github.com/DLR-RM/rl-baselines3-zoo/workflows/CI/badge.svg)
[![文档状态](https://readthedocs.org/projects/rl-baselines3-zoo/badge/?version=master)](https://rl-baselines3-zoo.readthedocs.io/en/master/?badge=master)
![覆盖率报告](https://img.shields.io/badge/coverage-68%25-brightgreen.svg?style=flat") [![代码风格](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# RL Baselines3 Zoo：稳定 Baselines3 强化学习代理的训练框架

<img src=“images/car.jpg”align=“右”宽度=“40％”/>

RL Baselines3 Zoo 是强化学习 (RL) 的训练框架，使用 [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)。

它提供用于训练、评估代理、调整超参数、绘制结果和录制视频的脚本。

此外，它还包括一系列针对常见环境和强化学习算法调整的超参数，以及使用这些设置进行训练的代理。


我们正在**寻找贡献者**来完成收藏！

该存储库的目标：

1. 提供简单的界面来训练和享受 RL 代理
2. 对不同的强化学习算法进行基准测试
3. 为每个环境和强化学习算法提供调整后的超参数
4. 与训练有素的特工一起享受乐趣！

这是原始 SB2 [rl-zoo](https://github.com/araffin/rl-baselines-zoo) 的 SB3 版本。

## 文档

文档可在线获取：[https://rl-baselines3-zoo.readthedocs.io/](ZXQMASK0X)

Ship3DOF 回购指南：

- 工作流程快速指南：[`scripts/ship_3dof/README.md`](scripts/ship_3dof/README.md)
- 扩展指南：[`docs/guide/ship_3dof.md`](docs/guide/ship_3dof.md)
- 中文操作手册：[`docs/guide/ship_3dof_user_manual_zh.md`](docs/guide/ship_3dof_user_manual_zh.md)

## 安装

### 最小化安装

来自来源：
```
pip install -e .
```

作为 python 包：
```
pip install rl_zoo3
```

注意：您可以从任何文件夹执行`python -m rl_zoo3.train`，并且可以访问`rl_zoo3`命令行界面，例如，`rl_zoo3 train`相当于`python train.py`

### 完整安装（带有额外的环境和测试依赖项）

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
pip install -e .[plots,tests]
```

请参阅 [稳定基线 3 文档](https://stable-baselines3.readthedocs.io/en/master/) 了解安装稳定基线 3 的替代方法。

## 培训代理人

每个环境的超参数在 `hyperparameters/algo_name.yml` 中定义。

如果此文件中存在环境，则您可以使用以下方法训练代理：
```
python train.py --algo algo_name --env env_id
```

每 10000 步评估一次代理，使用 10 个episode进行评估（仅使用一个评估环境）：
```
python train.py --algo sac --env HalfCheetahBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```

[文档](https://rl-baselines3-zoo.readthedocs.io) 中提供了更多示例。


## 集成

RL Zoo 与其他库/服务有一些集成，例如用于实验跟踪的权重和偏差或用于存储/共享训练模型的 Hugging Face。您可以在文档的[专用部分](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/integrations.html) 中找到更多信息。

## 情节脚本

请参阅文档的[专用部分](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/plot.html)。

## 享受训练有素的代理

**注意：要下载包含经过训练的代理的存储库，您必须使用 `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`** 才能克隆子模块。


如果经过训练的代理存在，那么您可以使用以下命令查看它的运行情况：
```
python enjoy.py --algo algo_name --env env_id
```

例如，在 5000 个时间步内享受 Breakout 上的 A2C：
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

## 超参数调优

请参阅文档的[专用部分](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html)。

## 自定义配置

请参阅文档的[专用部分](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/config.html)。

## 当前收藏：200 多名训练有素的特工！

经过训练的智能体的最终性能可以在[`benchmark.md`](./benchmark.md)中找到。要计算它们，只需运行 `python -m rl_zoo3.benchmark`。

受过训练的代理的列表和视频可以在我们的 Huggingface 页面上找到：https://huggingface.co/sb3

*注意：这不是定量基准，因为它仅对应于一次运行（参见 [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)）。该基准测试旨在检查算法（最大）性能，发现潜在错误，并允许用户访问预先训练的代理。*

### 雅达利游戏

来自 OpenAI 基准测试的 7 个 atari 游戏（NoFrameskip-v4 版本）。

|  强化学习算法 |  光束骑士 |突破|耐力赛|  乒乓球 |奎伯特 |海洋探索 |太空侵略者 |
|----------|--------------------|--------------------|--------------------|-------|-------|--------------------|--------------------|
| A2C | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

其他雅达利游戏（待完成）：

|  强化学习算法 |  帕克曼小姐 |小行星|走鹃 |
|----------|-------------|-----------|------------|
| A2C | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### 经典控制环境

|  强化学习算法 |  CartPole-v1 | MountainCar-v0 | Acrobot-v1 |摆锤-v1 | MountainCarContinously-v0 |
|----------|--------------|----------------|------------|--------------------|--------------------------|
|辅助研究系统 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| A2C | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |不适用 |不适用 |
| QR-DQN | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |不适用 |不适用 |
| DDPG|  不适用 |  不适用 |不适用 | :heavy_check_mark: | :heavy_check_mark: |
|国家安全委员会 |  不适用 |  不适用 |不适用 | :heavy_check_mark: | :heavy_check_mark: |
| TD3 |  不适用 |  不适用 |不适用 | :heavy_check_mark: | :heavy_check_mark: |
|全面质量控制 |  不适用 |  不适用 |不适用 | :heavy_check_mark: | :heavy_check_mark: |
| TRPO | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### Box2D 环境

|  强化学习算法 |  BipedalWalker-v3 |月球着陆器-v2 | LunarLander连续-v2 |  BipedalWalkerHardcore-v3 | BipedalWalkerHardcore-v3 | BipedalWalkerHardcore-v3 | BipedalWalkerHardcore-v3赛车-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
|辅助研究系统 |  | :heavy_check_mark: | | :heavy_check_mark: | |
| A2C | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DQN |不适用 | :heavy_check_mark: |不适用 |不适用 |不适用 |
| QR-DQN |不适用 | :heavy_check_mark: |不适用 |不适用 |不适用 |
| DDPG| :heavy_check_mark: |不适用 | :heavy_check_mark: | | |
|战略咨询委员会 | :heavy_check_mark: |不适用 | :heavy_check_mark: | :heavy_check_mark: | |
| TD3 | :heavy_check_mark: |不适用 | :heavy_check_mark: | :heavy_check_mark: | |
|全面质量控制 | :heavy_check_mark: |不适用 | :heavy_check_mark: | :heavy_check_mark: | |
| TRPO | | :heavy_check_mark: | :heavy_check_mark: | | |

### PyBullet 环境

见https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs.
与 [MuJoCo Envs](https://gym.openai.com/envs/#mujoco) 类似，但有一个~免费~（MuJoCo 2.1.0+ 现在免费！）易于安装的模拟器：pybullet。我们使用的是 `BulletEnv-v0` 版本。

注意：这些环境源自 [Roboschool](https://github.com/openai/roboschool)，并且比 Mujoco 版本更难（请参阅 [Pybullet 问题](https://github.com/bulletphysics/bullet3/issues/1718#issuecomment-393198883)）

|  强化学习算法 |  沃克二维|半猎豹 |蚂蚁 |雷彻 |  漏斗|人形 |
|----------|-----------|-------------|-----|---------|---------|----------|
|辅助研究系统 |  |  |  |  |  | |
| A2C | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
|战略咨询委员会 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TD3 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
|全面质量控制 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TRPO | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |

PyBullet 环境（续）

|  强化学习算法 |  迷你牛头怪 |迷你牛头鸭 |倒立双摆|倒立摆Swingup |
|----------|-----------|-------------|-----|---------|
| A2C | | | | |
|聚苯醚 | | | | |
| DDPG| | | | |
|国家安全委员会 | | | | |
| TD3 | | | | |
|全面质量控制 | | | | |

### 穆乔科环境

|  强化学习算法 |  沃克2d |半猎豹 |蚂蚁 |游泳运动员 |  漏斗|人形 |
|----------|-----------|-------------|-----|---------|---------|----------|
|辅助研究系统 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| A2C | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG|  |  |  |  |  | |
|国家安全委员会 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TD3 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|全面质量控制 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TRPO | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |

### 机器人环境

参见 https://gym.openai.com/envs/#robotics 和 https://github.com/DLR-RM/rl-baselines3-zoo/pull/71

MuJoCo版本：1.50.1.0
健身房版本：0.18.0

我们使用 v1 环境。

|  强化学习算法 |  获取Reach |获取拾取和放置 |获取推送 |获取幻灯片 |
|----------|-------------|-------------------|-----------|------------|
| HER+TQC | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### 熊猫机器人环境

见https://github.com/qgallouedec/panda-gym/.

与 [MuJoCo Robotics Envs](https://gym.openai.com/envs/#robotics) 类似，但有一个〜免费〜易于安装的模拟器：pybullet。

我们使用 v1 环境。

|  强化学习算法 |  熊猫到达 |熊猫取放 |熊猫推 |熊猫滑梯 |熊猫栈 |
|----------|-------------|-------------------|-----------|------------|------------|
| HER+TQC | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### 迷你电网环境

见https://github.com/Farama-Foundation/Minigrid.
著名网格世界的简单、轻量级和快速的 Gym 环境实现。

|强化学习算法 |空随机 5x5 |四房 | DoorKey-5x5 | 门钥匙多房间-N4-S5 |获取-5x5-N2 | GoToDoor-5x5 | PutNear-6x6-N2 |红蓝门-6x6 |上锁的房间 |钥匙走廊S3R1 |解锁| ObstructedMaze-2Dlh | 障碍迷宫
| ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------- |
| A2C |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |
|聚苯醚 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |
| QR-DQN |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |
| TRPO |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |

总共有 22 个环境组（每个环境组都有变体）。


## Colab 笔记本：在线试用！

您可以使用[Colab笔记本](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb)在线培训代理。

### 在交互式会话中传递参数

Zoo 并不意味着从交互式会话（例如：Jupyter Notebooks、IPython）中执行，但是，可以通过修改 `sys.argv` 并添加所需的参数来完成。

*例子*
```python
import sys
from rl_zoo3.train import train

sys.argv = ["python", "--algo", "ppo", "--env", "MountainCar-v0"]

train()
```


## 测试

要运行测试，首先安装 pytest，然后：
```
make pytest
```

与 pytype 的类型检查相同：
```
make type
```


## 引用该项目

要在出版物中引用此存储库：

```bibtex
@misc{rl-zoo3,
  author = {Raffin, Antonin},
  title = {RL Baselines3 Zoo},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
}
```

## 贡献

如果您训练的代理不存在于 RL Zoo 中，请提交 Pull 请求（也包含超参数和分数）。

## 贡献者

我们要感谢我们的贡献者：[@iandanforth](https://github.com/iandanforth)、[@tatsubori](https://github.com/tatsubori) [@Shade5](https://github.com/Shade5) [@mcres](https://github.com/mcres)、[@ernestum](https://github.com/ernestum)、[@qgallouedec](https://github.com/qgallouedec)
