（安装）=

# 安装

## 先决条件

RL Zoo 需要 Python 3.10+ 和 PyTorch >= 2.3

## 最小化安装

要使用 pip 安装 RL Zoo，请执行：

```bash
pip install rl_zoo3
```

来自来源：

```bash
git clone https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo/
pip install -e .
```

：：：{笔记}
您可以从任何文件夹执行`python -m rl_zoo3.train`，并且可以访问`rl_zoo3`命令行界面，例如，`rl_zoo3 train`相当于`python train.py`
:::

## 完整安装

使用额外的环境和测试依赖项：

：：：{笔记}
如果你想使用Atari游戏，你需要做`pip install "autorom[accept-rom-license]"`
另外还要下载 ROM
:::

```bash
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
pip install -e .[plots,tests]
```

请参阅 [稳定基线 3 文档](https://stable-baselines3.readthedocs.io/en/master/) 了解安装稳定基线 3 的替代方法。

## Docker 镜像

构建docker镜像（CPU）：

```
make docker-cpu
```

图形处理器：

```
USE_GPU=True make docker-gpu
```

拉取构建的 docker 镜像（CPU）：

```
docker pull stablebaselines/rl-baselines3-zoo-cpu
```

GPU图像：

```
docker pull stablebaselines/rl-baselines3-zoo
```

在 docker 镜像中运行脚本：

```
./scripts/run_docker_cpu.sh python train.py --algo ppo --env CartPole-v1
```
