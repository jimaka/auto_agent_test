---
神秘：
替换：
科拉布：|-
      ```{image} ../_static/img/colab.svg
      ```
---

（快速入门）=

# 入门

：：：{笔记}
您可以使用 Google Colab {{ Colab }} 在线尝试以下示例
笔记本：[RL Baselines 动物园笔记本]
:::

每个环境的超参数定义在
`hyperparameters/algo_name.yml`。

如果该文件中存在环境，则可以训练代理
使用：

```
python -m rl_zoo3.train --algo algo_name --env env_id
```

或者，如果您位于 RL Zoo3 文件夹中：

```
python train.py --algo algo_name --env env_id
```

例如（带有评估和检查点）：

```
python -m rl_zoo3.train --algo ppo --env CartPole-v1 --eval-freq 10000 --save-freq 50000
```

如果经过训练的代理存在，那么您可以使用以下命令查看它的运行情况：

```
python -m rl_zoo3.enjoy --algo algo_name --env env_id
```

例如，在 5000 个时间步内享受 Breakout 上的 A2C：

```
python -m rl_zoo3.enjoy --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

[rl基线动物园笔记本]：ZXQMASK0X
