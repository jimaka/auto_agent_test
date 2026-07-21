(sbx)=

# 稳定基线 Jax (SBX)

[Stable Baselines Jax (SBX)](https://github.com/araffin/sbx) 是 Jax 中 Stable-Baselines3 的概念验证版本。

与 SB3 相比，它提供的功能数量最少，但速度更快（高达 20 倍！）：<https://twitter.com/araffin2/status/1590714558628253698>

它还与 RL Zoo 兼容。
为此，您需要创建两个文件。

`train_sbx.py`：

```python
import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from sbx import DQN, PPO, SAC, TQC, DroQ


rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    train()
```

然后您可以调用 `python train_sbx.py --algo sac --env Pendulum-v1` 并使用 RL Zoo CLI。

`enjoy_sbx.py`：

```python
import rl_zoo3
import rl_zoo3.enjoy
from rl_zoo3.enjoy import enjoy
from sbx import DQN, PPO, SAC, TQC, DroQ


rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.enjoy.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    enjoy()
```
