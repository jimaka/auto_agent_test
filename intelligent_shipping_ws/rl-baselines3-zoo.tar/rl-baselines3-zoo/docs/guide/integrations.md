（积分）=

# 集成

## Huggingface 集线器集成

经过训练的代理的列表和视频可以在我们的 Huggingface 页面上找到：<https://huggingface.co/sb3>

将模型上传到集线器（与 `enjoy.py` 的语法相同）：

```
python -m rl_zoo3.push_to_hub --algo ppo --env CartPole-v1 -f logs/ -orga sb3 -m "Initial commit"
```

您可以选择自定义`repo-name`（默认：`{algo}-{env_id}`）
传递 `--repo-name` 参数。

从集线器下载模型：

```
python -m rl_zoo3.load_from_hub --algo ppo --env CartPole-v1 -f logs/ -orga sb3
```

## 实验跟踪

我们支持跟踪实验数据，例如学习曲线和
通过[权重和偏差](https://wandb.ai)设置超参数。

以下命令

```
python train.py --algo ppo --env CartPole-v1 --track --wandb-project-name sb3
```

在此产生一个跟踪实验
[网址](https://wandb.ai/openrlbenchmark/sb3/runs/1b65ldmh)。

要将标签添加到运行（例如 `optimized`），请使用参数
`--wandb-tags optimized`。
