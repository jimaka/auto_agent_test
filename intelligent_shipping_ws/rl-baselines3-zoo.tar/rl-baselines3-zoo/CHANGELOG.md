## 版本 2.9.1 (2026-06-15)

### 重大变化
- 升级到 SB3 >= 2.9.0，添加 pandas 到额外的“图”依赖项
- 放松的健身房版本范围（从 `"gymnasium>=0.29.1,<1.3.0"` 到 `"gymnasium>=0.29.1,<2.0"`）

### 新功能

### 错误修复

### 文档

### 其他


## 版本 2.8.0 (2026-04-01)

### 重大变化
- 升级至 SB3 >= 2.8.0
- 删除了对 Python 3.9 的支持，请升级到 Python >= 3.10
- 设置“`strict=True`` for every call to ``zip(...)`”

### 新功能
- 添加了对Python 3.13的官方支持
- 允许在 hyperparam 配置中指定 `env_kwargs`
- 允许在任何环境中使用默认超参数
- 在权重和偏差 (Wandb) 中保存训练命令
- 将训练命令和默认超参数保存为研究属性

### 错误修复

### 文档
- 切换到 Markdown 文档（使用 MyST 解析器）

### 其他
- 修复了 `plot_from_file.py` 中未使用的变量

## 版本 2.7.0 (2025-07-25)

### 重大变化
- 升级至 SB3 >= 2.7.0
- `linear_schedule` 现在返回 `SimpleLinearSchedule` 对象以实现更好的可移植性
- 在超参数中将 `LunarLander-v2` 重命名为 `LunarLander-v3`
- 在超参数中将 `CarRacing-v2` 重命名为 `CarRacing-v3`

### 新功能
- 添加了 Gymnasium v​​1.2 支持

### 错误修复
- Docker GPU 镜像现在又可以工作了
- 使用`ConstantSchedule`和`SimpleLinearSchedule`代替`constant_fn`和`linear_schedule`
- 修复了较新 Gymnasium 版本的 `CarRacing-v3` 超参数

## 版本2.6.0 (2025-03-24)

### 重大变化
- 升级到SB3 >= 2.6.0
- 重构超参数优化。现在支持 Optuna [日志存储后端](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html)（推荐默认值），您可以通过 `train.py` 的新 `--trial-id` 参数轻松加载调整后的超参数。

例如，使用日志存储进行优化：
```bash
python train.py --algo ppo --env Pendulum-v1 -n 40000 --study-name demo --storage logs/demo.log --sampler tpe --n-evaluations 2 --optimize --no-optim-plots
```
使用 [optuna-dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html) 进行实时可视化
```
optuna-dashboard logs/demo.log
```

从试验编号 21 加载超参数并用它训练代理：
```bash
python train.py --algo ppo --env Pendulum-v1 --study-name demo --storage logs/demo.log --trial-id 21
```


### 新功能
- 保存用于启动训练的确切命令行
- 通过允许覆盖 `VecEnv` 类来实例化 `ExperimentManager` 中的 env，添加了对特殊矢量化 env（例如 Brax、IsaacSim）的支持
- 允许通过传递 `--log-interval -2` 来禁用自动日志记录（在手动记录日志时很有用）
- 添加了 Gymnasium v​​1.1 支持

### 错误修复
- 修复了 `get_hf_trained_models()` 中旧 HF api 的使用

### 文档

### 其他
- 由于新的超参数优化脚本，`scripts/parse_study.py` 现已弃用

## 版本 2.5.0 (2025-01-27)

### 重大变化
- 升级到 Pytorch >= 2.3.0
- 升级至 SB3 >= 2.5.0

### 新功能
- 添加了对 Numpy v2 的支持
- 添加了对将回调和 env 包装器指定为 python 配置文件中的 python 对象（而不是字符串）的支持

### 错误修复

### 文档

### 其他
- 更新的 Dockerfile

## 版本2.4.0 (2024-11-18)

**新算法：CrossQ、Gymnasium v​​1.0 支持，以及 Swimmer-v4 env 上 SAC/TQC 的更好默认值**

### 重大变化
- 更新了 Swimmer-v4 的 TQC/SAC 的默认超参数（减少 gamma 以获得更一致的结果）(@JacobHA) [W&B 报告](https://wandb.ai/openrlbenchmark/sbx/reports/SAC-MuJoCo-Swimmer-v4--Vmlldzo3NzM5OTk2)
- 升级到SB3 >= 2.4.0
- 在超参数中将 `LunarLander-v2` 重命名为 `LunarLander-v3`

### 新功能
- 为 SB3-contrib 添加了 `CrossQ` 超参数 (@danielpalen)
- 添加了 Gymnasium v​​1.0 支持

### 错误修复
- 当推送到 Hugging Face Hub 时，用推荐的 `HfApi` 替换了已弃用的 `huggingface_hub.Repository`（请参阅 https://huggingface.co/docs/huggingface_hub/concepts/git_vs_http) (@cochaviz)

### 文档

### 其他
- CI 中的 PyTorch 版本更新为 2.4.1
- 切换到 uv 以更快地在 GitHub CI 上下载包

## 版本2.3.0 (2024-03-31)

### 重大变化
- 更新了 TD3/DDPG 的默认超参数，使其与 SAC 更加一致
- 将 MuJoCo envs 超参数升级到 v4（需要更新预训练代理）
- 升级到SB3 >= 2.3.0

### 其他
- 添加了对 `setup.py` 的测试依赖项 (@power-edge)
- 简化`requirements.txt`的依赖关系（从`setup.py`中删除重复项）


## 版本2.2.1 (2023-11-17)

### 重大变化
- 删除了 `gym` 依赖项，某些预训练代理仍然需要该包。
- 升级至SB3 >= 2.2.1
- 升级至 Huggingface-SB3 >= 3.0
- 升级到 pytablewriter >= 1.0

### 新功能
- 将 `--eval-env-kwargs` 添加到 `train.py` (@Quentin18)
- 将 `ppo_lstm` 添加到 hyperparams_opt.py (@technocrat13)

### 错误修复
- 升级至`pybullet_envs_gymnasium>=0.4.0`
- 删除了旧的黑客（例如在测试时将离策略算法限制为一个环境）

### 文档

### 其他
- 更新了 docker 镜像，删除了对 X 服务器的支持
- 将已弃用的 `optuna.suggest_uniform(...)` 替换为 `optuna.suggest_float(..., low=..., high=...)`
- 切换到领口以对进口商品进行排序
- 更新了测试以使用 `shlex.split()`
- 修复了 `rl_zoo3/hyperparams_opt.py` 类型提示
- 修复了 `rl_zoo3/exp_manager.py` 类型提示

## 版本2.1.0 (2023-08-17)

### 重大变化
- 放弃了 python 3.7 支持
- SB3 现在需要 PyTorch 1.13+
- 升级到SB3 >= 2.1.0
- 升级到 Huggingface-SB3 >= 2.3
- 升级到 Optuna >= 3.0
- 升级到cloudpickle >= 2.2.1

### 新功能
- 添加了 python 3.11 支持

### 错误修复

### 文档

### 其他


## 版本2.0.0 (2023-06-22)

**体育馆支持**

> **警告**
> Stable-Baselines3 (SB3) v2.0.0 将是最后一个支持 python 3.7 的版本

### 重大变化
- 修复了 HistoryWrapper 中的错误，现在返回正确的 obs 空间限制
- 升级至 SB3 >= 2.0.0
- 升级到 Huggingface-SB3 >= 2.2.5
- 升级到 Gym API 0.26+，RL Zoo3 不再与 Gym 0.21 一起使用

### 新功能
- 添加了体育馆支持
- Gym 0.26+ 补丁以继续使用 pybullet 和 TimeLimit 包装器

### 错误修复
- 在超参数中将 `CarRacing-v1` 重命名为 `CarRacing-v2`
- Huggingface 推送到集线器现在接受 `--n-timesteps` 参数来调整视频的长度
- 修复了 `record_video` 步骤（在进入封闭环境之前）

## 版本1.8.0 (2023-04-07)

**新文档，多环境 HerReplayBuffer**

> **警告**
> Stable-Baselines3 (SB3) v1.8.0 将是最后一个使用 Gym 作为后端的版本。
从 v2.0.0 开始，Gymnasium 将成为默认后端（尽管 SB3 将为 Gym env 提供兼容层）。
您可以在[此处](https://gymnasium.farama.org/content/migration-guide/)找到迁移指南。
如果您想尝试 SB3 v2.0 alpha 版本，可以查看 [PR #1327](https://github.com/DLR-RM/stable-baselines3/pull/1327)。

### 重大变化
- 升级至 SB3 >= 1.8.0
- 升级到支持多个环境的新 `HerReplayBuffer` 实现
- 删除了 Panda 和 Fetch env 的 `TimeFeatureWrapper`，因为新的重播缓冲区应该处理超时。

### 新功能
- Swimmer 上 RecurrentPPO 的调整超参数
- 文档现在使用 Sphinx 构建并托管在阅读文档上
- 在 11 个 MiniGrid 环境上为 PPO 添加了超参数预训练代理

### 错误修复
- 为 CI 设置 ``highway-env`` version to 1.5 and ``setuptools to`` v65.5
- 删除了用于推送到集线器实用程序的 `use_auth_token`
- HumanoidStandup、Reacher、InvertedPendulum 和 InvertedDoublePendulum 从 v3 恢复到 v2，因为它们不是 mujoco 重构的一部分（请参阅 https://github.com/openai/gym/pull/1304)
- 修复了 `gym-minigrid` 策略（从 `MlpPolicy` 到 `MultiInputPolicy`）

### 文档

### 其他
- 在 Makefile 中添加了对 `ruff`（flake8 的快速替代品）的支持
- 删除了 Gitlab CI 文件
- 将已弃用的 `optuna.suggest_loguniform(...)` 替换为 `optuna.suggest_float(..., log=True)`
- 切换为`ruff`和`pyproject.toml`
- 使用 `HerReplayBuffer` 时删除了 `online_sampling` 和 `max_episode_length` 参数

## 版本1.7.0 (2023-01-10)

**SB3 v1.7.0，添加了对 python 配置文件的支持**

### 重大变化
- `--yaml-file` 参数已重命名为 `-conf` (`--conf-file`)，因为现在也支持 python 文件
- 升级至SB3 >= 1.7.0（将`net_arch=[dict(pi=.., vf=..)]`更改为`net_arch=dict(pi=.., vf=..)`）

### 新功能
- 现在支持在 yaml 文件中指定自定义策略 (@Rick-v-E)
- 添加了“`monitor_kwargs`”参数
- 在 `enjoy` 回放中处理 panda-gym v1 envs 的 `render:True` 的 `env_kwargs` 以匹配其他 envs 的可视化行为
- 添加了对 python 配置文件的支持
- Swimmer 上 PPO 的超参数调整
- 添加了 ``-tags/--wandb-tags`` argument to ``train.py`` 以将标签添加到 wandb 运行
- 向 wandb 运行添加了 sb3 版本标签

### 错误修复
- 允许直接调用`python -m rl_zoo3.cli`
- 修复了使用子进程时尽管传递“`--gym-package`”但未找到自定义环境的错误
- 修复了 MinitaurBulletEnv-v0、MinitaurBulletDuckEnv-v0、HumanoidBulletEnv-v0、InvertedDoublePendulumBulletEnv-v0 和 InvertedPendulumSwingupBulletEnv 的 TRPO 超参数

### 文档

### 其他
- `scripts/plot_train.py` 绘制模型，使新模型出现在旧模型之上。
- 使用 mypy 添加了额外的类型检查
- 标准化“`from gym import spaces`”的使用


## 版本1.6.3 (2022-10-13)

### 重大变化

### 新功能

### 错误修复
- `python3 -m rl_zoo3.train` 现在按预期工作

### 文档
- 添加了有关在交互式会话中传递参数的说明和示例 (@richter43)

### 其他
- 使用问题表单而不是问题模板


## 版本 1.6.2.post2 (2022-10-10)

### 重大变化
- RL Zoo 现在是一个 python 包
- 低通滤波器被移除
- 升级到稳定基线3 (SB3) >= 1.6.2
- 升级到 sb3-contrib >= 1.6.2
- 现在使用内置SB3 `ProgressBarCallback`代替`TQDMCallback`

### 新功能
- RL Zoo cli：`rl_zoo3 train` 和 `rl_zoo3 enjoy`

### 错误修复

### 文档

### 其他

## 版本1.6.1 (2022-09-30)

**进度条和自定义yaml文件**

### 重大变化
- 升级到稳定基线3 (SB3) >= 1.6.1
- 升级到 sb3-contrib >= 1.6.1

### 新功能
- 为 `train.py` 添加了 `--yaml-file` 参数选项，以从自定义 yaml 文件中读取超参数 (@JohannesUl)

### 错误修复
- 在 record_video.py 上添加了 `custom_object` 参数 (@Affonso-Gui)
- 将 record_video.py 上 DQN/QR-DQN 的 `optimize_memory_usage` 更改为 `False` (@Affonso-Gui)
- In `ExperimentManager` `_maybe_normalize` set `training` to `False` for eval envs,
防止标准化统计数据在 eval envs 中更新（例如在 EvalCallback 中）(@pchalasani)。
- 在优化超参数时仅使用一个 env 来获取操作空间，并且它被正确关闭（@SammyRamone）
- 使用 tqdm 和 rich 通过 `-P` 参数添加进度条

### 文档

### 其他

## 版本1.6.0 (2022-08-05)

**RecurrentPPO (ppo_lstm) 和 Huggingface 集成**

### 重大变化
- 将超参数优化试验数量的默认值从 10 更改为 500。(@ernestum)
- 从时间步数导出中间修剪评估的数量（每 100k 时间步 1 次评估。）(@ernestum)
- 将默认 --eval-freq 从 10k 步更新为 25k 步
- 将 `HistoryWrapper` 的默认范围更新为 2
- 升级到稳定基线3 (SB3) >= 1.6.0
- 升级到 sb3-contrib >= 1.6.0

### 新功能
- 支持使用 `--device` 标志设置 PyTorch 的设备 (@gregwar)
- 添加`--max-total-trials`参数以帮助分布式优化。 (@ernestum)
- 在配置中添加了 `vec_env_wrapper` 支持（与 `env_wrapper` 工作方式相同）
- 添加了 Huggingface 集线器集成
- 添加了 `RecurrentPPO` 支持（又名 `ppo_lstm`）
- 添加了从中心自动下载“官方”sb3 型号的功能
- 为 A2C 添加了 Humanoid-v3、Ant-v3、Walker2d-v3 模型（@pseudo-rnd-thoughts）
- 添加了 MsPacman 模型

### 错误修复
- 修复 PPO 超参数文件中的 `Reacher-v3` 名称
- 固定 ale-py==0.7.4 直到新的 SB3 版本发布
- 使用 LSTM 策略修复欣赏/录制视频的问题
- 修复名称中带有斜杠的环境的错误 (@ernestum)
- 将 Atari 游戏上的 DQN/QR-DQN 的 `optimize_memory_usage` 更改为 `False`，
如果你想节省RAM，你需要停用`handle_timeout_termination`
在`replay_buffer_kwargs`中

### 文档

### 其他
- 当 pruner 设置为 `"none"` 时，使用 `NopPruner` 而不是转向的 `MedianPruner` (@qgallouedec)

## 版本1.5.0 (2022-03-25)

**支持权重和偏差实验跟踪**

### 重大变化
- 升级到稳定基线3 (SB3) >= 1.5.0
- 升级到 sb3-contrib >= 1.5.0
- 升级到健身房0.21

### 新功能
- 现在可以使用调试模式激活每个试验的详细模式（在进行超参数优化时）（详细 == 2）
- 通过 `--track` 标志 (@vwxyzjn) 支持通过权重和偏差进行实验跟踪
- 支持通过 `RawStatisticsCallback` 跟踪原始情景统计数据（@vwxyzjn，请参阅 https://github.com/DLR-RM/rl-baselines3-zoo/pull/216)

### 错误修复
- 在新系统上使用分布式 Optuna 负载进行优化期间保存的策略 (@jkterry)
- 修复了录制视频的脚本与享受脚本不同步的问题

### 文档

### 其他

## 版本1.4.0 (2022-01-19)

### 重大变化
- 放弃了 python 3.6 支持
- 升级到稳定基线3 (SB3) >= 1.4.0
- 升级到 sb3-contrib >= 1.4.0

### 新功能
- 添加了 mujoco 超参数
- 添加了 MuJoCo 预训练代理
- 添加脚本来解析 optuna 研究的最佳超参数
- 添加了 TRPO 支持
- 添加了 ARS 支持和预训练代理

### 错误修复

### 文档
- 替换正面图像

### 其他


## 版本1.3.0 (2021-10-23)

**可靠的情节和错误修复**

**警告：此版本将是支持 Python 3.6 的最后一个版本（将于 2021 年 12 月终止）。我们强烈建议您升级到 Python >= 3.7。**

### 重大变化
- 升级至panda-gym 1.1.1
- 升级到稳定基线3 (SB3) >= 1.3.0
- 升级到 sb3-contrib >= 1.3.0

### 新功能
- 添加了对使用 rliable 进行性能比较的支持

### 错误修复
- 使用 Dict obs 修复训练并通道最后的图像

### 文档

### 其他
- 更新了 docker 镜像
- 受限gym版本：gym>=0.17,<0.20
- Pendulum 上 A2C/PPO 更好的超参数

## 版本1.2.0 (2021-09-08)

### 重大变化
- 升级到稳定基线3 (SB3) >= 1.2.0
- 升级到 sb3-contrib >= 1.2.0

### 新功能
- 添加了对 Python 3.10 的支持

### 错误修复
- 修复 `--load-last-checkpoint` (@SammyRamone)
- 修复 `ExperimentManager` 中 `gym.Env` 类入口点的 `TypeError` (@schuderer)
- 修复超参数优化期间回调的使用（@SammyRamone）

### 文档

### 其他
- 将 python 3.9 添加到 Github CI
- 增加了 Atari 游戏的 DQN 重播缓冲区大小 (@nikhilrayaprolu)

## 版本1.1.0 (2021-07-01)

### 重大变化
- 升级到稳定基线3 (SB3) >= 1.1.0
- 升级到 sb3-contrib >= 1.1.0
- 添加超时处理（参见 SB3 文档）
- `HER` 现在是一个重播缓冲区类，而不再是一种算法
- 删除了 `PlotNoiseRatioCallback`
- 删除了 `PlotActionWrapper`
- 将 Optuna 参数字典中的 `'lr'` 键更改为 `'learning_rate'`，以便该字典可以直接传递给 SB3 方法 (@jkterry)

### 新功能
- 添加对录制最佳模型和检查点视频的支持 (@mcres)
- 添加对录制训练实验视频的支持 (@mcres)
- 添加对字典观察的支持
- 添加实验性并行训练（使用 `utils.callbacks.ParallelTrainCallback`）
- 添加了对使用多个环境进行评估的支持
- 为享受脚本添加了 `--load-last-checkpoint` 选项
- 在超参数优化结束时保存 Optuna 研究对象并绘制结果（需要 `plotly` 软件包）
- 允许将多个文件夹传递给 `scripts/plot_train.py`
- 标记以保存每次训练运行的日志和最佳策略 (@jkterry)

### 错误修复
- 修复了 Linux 上 PyBullet env 的视频渲染
- 修复了 `get_latest_run_id()`，使其在 Windows 中也能工作 (@NicolasHaeffner)
- 修复了使用 `HER` 重播缓冲区时的视频录制

### 文档
- 更新了自述文件（现在支持 dict obs）

### 其他
- 将 `is_bullet()` 添加到 `ExperimentManager`
- 简化享受脚本的 `close()`
- 更新了 docker 镜像以包含最新的黑色版本
- 更新了 TD3 Walker2D 模型（感谢 @modanesh）
- 修复了情节标题中的拼写错误（@scotemmons）
- 添加到 `requirements.txt` 的最低 cloudpickle 版本 (@amy12xx)
- 修复了 atari-py 版本（最新版本中缺少 ROM）
- 更新了 `SAC` 和 `TD3` 搜索空间
- 清理 eval_freq 文档和变量名称更改 (@jkterry)
- 在优化过程中打印保存的超参数时添加澄清的打印语句（@jkterry）
- 澄清 n_evaluations 帮助文本 (@jkterry)
- 使用默认值的简化超参数文件
- 添加了新的 TQC+HER 代理
- 添加 `panda-gym` 环境 (@qgallouedec)

## 1.0版本（2021-03-17）

### 重大变化
- 升级到 SB3 >= 1.0
- 升级到 sb3-contrib >= 1.0

### 新功能
- 添加了 100 多个训练有素的代理 + 基准文件
- 添加对在 python 3.8+ 下加载已保存模型的支持（无法重新训练）
- 添加了机器人预训练代理（@sgillen）

### 错误修复
- `HER` 处理动作噪音的错误修复
- 修复了 `HER` 的双重重置错误并享受脚本

### 文档
- 添加了有关绘制脚本的文档

### 其他
- 更新了 `HER` 超参数

## 预发布 0.11.1 (2021-02-27)

### 重大变化
- 删除了 `LinearNormalActionNoise`
- 默认情况下，评估是确定性的，Atari 游戏除外
- 现在需要 `sb3_contrib`
- `TimeFeatureWrapper` 已移至 contrib 存储库
- 用更新的 `plot_training_success.py` 替换了旧的 `plot_train.py` 脚本
- 重命名了“`n_episodes_rollout`` to ``train_freq`”元组以匹配最新版本的 SB3

### 新功能
- 添加了选择用于多处理的 `VecEnv` 类的选项
- 添加了对 `TQC` 的超参数优化支持
- 添加了对 SB3 contrib 中的 `QR-DQN` 的支持

### 错误修复
- 改进了对 Atari 游戏的检测
- 修复没有足够时间步长时绘图脚本中的潜在错误
- 修复了使用 HER + DQN/TQC 进行超参数优化时的错误

### 文档
- 改进的文档 (@cboettig)

### 其他
- 重构训练脚本，现在使用 `ExperimentManager` 类
- 将 `make_env` 替换为 SB3 内置 `make_vec_env`
- 添加更多类型提示（`utils/utils.py` 完成）
- 尽可能使用 f 字符串
- 更改了 `PPO` atari 超参数（删除了 vf 剪辑）
- 更改了 `A2C` atari 超参数（优化器的 eps 值）
- 更新了基准测试脚本
- 更新了超参数优化搜索空间（针对 A2C/PPO 注释了 gSDE）
- 更新了 CartPole 的 `DQN` 超参数
- 不要包装通道优先图像环境（现在由 SB3 原生支持）
- 删除了记录成功率的黑客行为
- 简化剧情脚本

## 预发布0.10.0 (2020-10-28)

### 重大变化

### 新功能
- 添加了对 `HER` 的支持
- 在 `utils/wrappers.py` 中添加低通滤波器包装器
- 添加了 `TQC` 支持，来自 sb3-contrib 的实现

### 错误修复
- 修复了 `TimeFeatureWrapper` 推断最大时间步长的问题
- 修复了最新 Gym 版本的“`flatten_dict_observations`` in `utils/utils.py”（@ManifoldFR）
- `VecNormalize` 现在考虑 `gamma` 超参数
- 修复继续训练或使用经过训练的代理时 `VecNormalize` 的加载

### 文档

### 其他
- 添加了对包装器的测试
- 更新了绘图脚本


## 版本0.8.0 (2020-08-04)

### 重大变化

### 新功能
- 分布式优化（@SammyRamone）
- 添加了“`--load-checkpoints`”来加载特定检查点
- 添加了``--num-threads``来享受脚本
- 添加了 DQN 支持
- 添加了命令行参数的保存（@SammyRamone）
- 添加了 DDPG 支持
- 新增版本
- 添加了“`RMSpropTFLike`”支持

### 错误修复
- 修复 optuna 警告 (@SammyRamone)
- 修复了未考虑并行环境的 `--save-freq`
- 测试离策略模型（例如 SAC/DQN）时将 `buffer_size` 设置为 1，以避免内存分配问题
- 修复了 `enjoy.py` 加载时的种子
- 在 Atari 游戏上进行超参数优化时的非确定性评估
- 使用“最大化”进行超参数优化（@SammyRamone）
- 修复了进行超参数优化时奖励未标准化的错误 (@caburu)
- 对于 `MountainCar-v0` 和 `Acrobot-v1`，从 `ppo.yml` 中删除了 `nminibatches`。 （@blurLake）
- 修复 `--save-replay-buffer` 以兼容最新的 SB3 版本
- 训练结束时的封闭环境
- 更新了更简单的gym环境上的DQN超参数（由于实现中的更新）

### 文档

### 其他
- 重新格式化 `enjoy.py`、`test_enjoy.py`、`test_hyperparams_opt.py`、`test_train.py`、`train.py`、`callbacks.py`、`hyperparams_opt.py`、`utils.py`、`wrappers.py` (@salmannotkhan)
- 重新格式化 `record_video.py` (@salmannotkhan)
- 使用 flake8 添加了代码风格检查 `make lint`
- 重新格式化 `benchmark.py` (@salmannotkhan)
- 添加了 github ci
- 修复了大多数 linter 警告
- 现在使用黑色和 isort 进行自动格式化
- 更新了地块
