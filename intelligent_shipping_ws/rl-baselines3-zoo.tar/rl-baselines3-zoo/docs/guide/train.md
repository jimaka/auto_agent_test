（火车）=

# 培训代理

## 基本用法

每个环境的超参数定义在
`hyperparameters/algo_name.yml`。

：：：{笔记}
安装 RL Zoo3 后，您可以从任何文件夹运行 `python -m rl_zoo3.train`，这相当于 `python train.py`。
:::

如果此文件中存在环境，则您可以使用以下方法训练代理：

```
python train.py --algo algo_name --env env_id
```

：：：{笔记}
您可以使用 `-P` (`--progress`) 选项来显示进度条。
:::

## 自定义配置文件

当自定义配置文件是包含 `env_id` 条目的 YAML 文件时，请使用自定义配置文件：

```
python train.py --algo algo_name --env env_id --conf-file my_yaml.yml
```

您还可以使用包含名为 `hyperparams` 的字典的 python 文件，其中每个 `env_id` 都有一个条目。
（参见 `hyperparams/python/ppo_config_example.py` 示例）

```
# You can pass a path to a python file
python train.py --algo ppo --env MountainCarContinuous-v0 --conf-file hyperparams/python/ppo_config_example.py
# Or pass a path to a file from a module (for instance my_package.my_file)
python train.py --algo ppo --env MountainCarContinuous-v0 --conf-file hyperparams.python.ppo_config_example
```

这种方法的优点是可以指定任意的Python字典
并确保它们的所有依赖项都导入到配置文件本身中。

## 张量板、检查点、评估

例如（有张量板支持）：

```
python train.py --algo ppo --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/
```

每 10000 步评估一次代理，使用 10 集进行评估（仅使用一个评估环境）：

```
python train.py --algo sac --env AntBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```

每 100000 步保存一个代理的检查点：

```
python train.py --algo td3 --env AntBulletEnv-v0 --save-freq 100000
```

## 恢复培训

继续训练（此处，加载预训练的 Breakout 代理并继续训练 5000 步）：

```
python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i rl-trained-agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000
```

## 保存重播缓冲区

使用离策略算法时，您还可以在训练后**保存重播缓冲区**：

```
python train.py --algo sac --env Pendulum-v1 --save-replay-buffer
```

如果存在，继续训练时将自动加载。

## Env 关键字参数

您可以指定关键字参数传递给 env 构造函数
命令行，使用 `--env-kwargs`：

```
python enjoy.py --algo ppo --env MountainCar-v0 --env-kwargs goal_velocity:10
```

## 覆盖超参数

您可以使用以下命令轻松地在命令行中覆盖超参数
`--hyperparams`：

```
python train.py --algo a2c --env MountainCarContinuous-v0 --hyperparams learning_rate:0.001 policy_kwargs:"dict(net_arch=[64, 64])"
```

注意：如果你想传递一个字符串，你需要像这样转义它：
ZXQ掩码0X
