（享受）=

# 享受训练有素的代理

：：：{笔记}
要使用经过培训的代理下载存储库，您必须使用
ZXQ掩码0X
为了也克隆子模块。
:::

## 享受训练有素的代理

如果经过训练的代理存在，那么您可以使用以下命令查看它的运行情况：

```
python enjoy.py --algo algo_name --env env_id
```

例如，在 5000 个时间步内享受 Breakout 上的 A2C：

```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

如果您自己培训过代理，您需要执行以下操作：

```
# exp-id 0 corresponds to the last experiment, otherwise, you can specify another ID
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 0
```

## 加载检查点，最佳模型

加载最佳模型（使用评估环境时）：

```
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-best
```

加载检查点（这里检查点名称是
`rl_model_10000_steps.zip`）：

```
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-checkpoint 10000
```

加载最新的检查点：

```
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-last-checkpoint
```

## 录制训练有素的特工的视频

使用最新保存的模型记录 1000 步：

```
python -m rl_zoo3.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000
```

使用保存的最佳模型来代替：

```
python -m rl_zoo3.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000 --load-best
```

录制训练期间保存的检查点的视频（此处为
检查点名称为 `rl_model_10000_steps.zip`）：

```
python -m rl_zoo3.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000 --load-checkpoint 10000
```

## 录制训练实验视频

除了录制特定保存模型的视频外，它还可以
可以录制检查点的训练实验视频
已获救。

每个检查点记录 1000 步，最新和最好保存的模型：

```
python -m rl_zoo3.record_training --algo ppo --env CartPole-v1 -n 1000 -f logs --deterministic
```

上一条命令将创建一个 `mp4` 文件。将此文件转换为
`gif` 格式也是：

```
python -m rl_zoo3.record_training --algo ppo --env CartPole-v1 -n 1000 -f logs --deterministic --gif
```
