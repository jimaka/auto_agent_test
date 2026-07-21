（调整）=

# 超参数调优

## 自动超参数优化

博文：[自动超参数调优 - 视觉指南](https://araffin.github.io/post/hyperparam-tuning/)

视频：<https://www.youtube.com/watch?v=AidFTOdGNFQ>

我们使用[Optuna](https://optuna.org/)来优化
超参数。 Not all hyperparameters are tuned, and tuning enforces
某些默认超参数设置可能与
官方默认。看
[rl_zoo3/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py)
每个代理的当前设置。

中未指定的超参数
[rl_zoo3/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py)
取自关联的 YAML 文件并回退到默认值
SB3 的值（如果不存在）。

注意：使用SuccessiveHalvingPruner（“减半”）时，必须指定
ZXQ掩码0X

1000 次试验的预算，最多 50000 个步骤：

```
python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
  --sampler tpe --pruner median
```

使用共享数据库的分布式优化也是可能的（参见
相应的[Optuna
文档](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)):

```
python train.py --algo ppo --env MountainCar-v0 -optimize --study-name test --storage logs/demo.log
```

使用 [optuna-dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html) 进行实时可视化

```bash
optuna-dashboard logs/demo.log
```

从试验编号 21 加载超参数并用它训练代理：

```bash
python train.py --algo ppo --env MountainCar-v0 --study-name test --storage logs/demo.log --trial-id 21
```

超参数调整的默认预算是 500 次试验，并且有
每 10 万次进行一次修剪/提前停止的中间评估
步骤。

## 超参数搜索空间

请注意，调整时动物园中使用的默认超参数是
并不总是与中提供的默认值相同
[稳定基线3](https://stable-baselines3.readthedocs.io/en/master/modules/base.html)。
请查阅最新的源代码以确保这些设置。为了
例子：

- PPO 调整采用 `ortho_init = False` 的网络架构
调谐时，虽然是`True`
[默认](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#ppo-policies)。
您可以通过更新来更改它
[rl_zoo3/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py)。
- TD3 和 DDPG 中的非间歇性推出假设
`gradient_steps = train_freq`，因此仅调整 `train_freq`
减少搜索空间。
