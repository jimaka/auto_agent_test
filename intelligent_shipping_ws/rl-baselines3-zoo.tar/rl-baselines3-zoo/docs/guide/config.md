（配置）=

# 配置

## 超参数 YAML 语法

`hyperparameters/algo_name.yml`中用于设置的语法
超参数（同样的语法是[覆盖
超参数](https://github.com/DLR-RM/rl-baselines3-zoo#overwrite-hyperparameters)
如果参数是函数，则在 cli 上）可能会被特化。看
`hyperparameters/` 目录中的示例。例如：

- 指定学习率的线性计划：

```yaml
learning_rate: lin_0.012486195510232303
```

为网络指定不同的激活函数：

```yaml
policy_kwargs: "dict(activation_fn=nn.ReLU)"
```

对于自定义策略：

```yaml
policy: my_package.MyCustomPolicy  # for instance stable_baselines3.ppo.MlpPolicy
```

## 环境标准化

超参数文件中，`normalize: True`表示训练
环境将被包裹在
[向量标准化](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py#L13)
包装纸。

[正常化
使用](https://github.com/DLR-RM/rl-baselines3-zoo/issues/64)
`VecNormalize`的默认参数，`gamma`除外
它被设置为与代理的相匹配。这可以是
[覆盖](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v0.10.0/hyperparams/sac.yml#L239)
使用适当的 `hyperparameters/algo_name.yml`，例如

```yaml
normalize: "{'norm_obs': True, 'norm_reward': False}"
```

## 环境包装器

您可以在超参数配置中指定要使用的一个或多个包装器
周围环境：

对于一个包装纸：

```yaml
env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
```

对于多个，指定一个列表：

```yaml
env_wrapper:
    - rl_zoo3.wrappers.TruncatedOnSuccessWrapper:
        reward_offset: 1.0
    - sb3_contrib.common.wrappers.TimeFeatureWrapper
```

请注意，您也可以轻松指定参数。

默认情况下，环境使用 `Monitor` 包装器进行包装
记录剧集统计数据。您可以使用指定参数
用于记录附加数据的 `monitor_kwargs` 参数。该数据*必须*是
出现在每集最后一步的信息词典中。

例如，用于记录目标环境的成功
（例如 `FetchReach-v1`）：

```yaml
monitor_kwargs: dict(info_keywords=('is_success',))
```

或使用 `Ant-v3` 记录最终 x 位置：

```yaml
monitor_kwargs: dict(info_keywords=('x_position',))
```

注：对于已知的`GoalEnv`，如`FetchReach`，
`info_keywords=('is_success',)` 实际上是默认值。

您还可以使用以下方式指定环境关键字参数：

```yaml
env_kwargs:
  gravity: 0.0
```

## VecEnv包装器

可以在config中指定使用哪个`VecEnvWrapper`，同理
与 env 包装器一样（见上文），使用 `vec_env_wrapper` 键：

例如：

```yaml
vec_env_wrapper: stable_baselines3.common.vec_env.VecMonitor
```

注：使用`normalize`单独支持`VecNormalize`
关键字，而`VecFrameStack`有一个专用关键字`frame_stack`。

## 回调

遵循与 env 包装器相同的语法，您还可以添加自定义
训练期间使用的回调。

```yaml
callback:
  - rl_zoo3.callbacks.ParallelTrainCallback:
      gradient_steps: 256
```

## 默认超参数

您可以在超参数 YAML 文件中使用 `default` 条目，为没有特定条目的环境提供后备超参数。
当在没有调整超参数的环境中进行训练时，这非常有用。

`default` 超参数将在以下情况下使用：
1. 配置文件中未明确列出环境
2. 环境不是 Atari 游戏（使用 `atari` 条目）

例子：

```yaml
# Specific hyperparameters for CartPole-v1
CartPole-v1:
  n_envs: 8
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  learning_rate: 1e-3

# Fallback hyperparameters for any other environment
default:
  n_envs: 4
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
```

当在未明确列出的环境中进行训练时，Zoo 将打印 `Using 'default' hyperparameters` 并应用默认设置。
