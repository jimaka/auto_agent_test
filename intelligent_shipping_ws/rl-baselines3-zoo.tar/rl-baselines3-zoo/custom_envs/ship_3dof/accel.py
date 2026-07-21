"""训练加速回调：通过 rl_zoo3 的 `callback` 超参注入。

用法（hyperparams 或 --hyperparams）:
    callback:custom_envs.ship_3dof.accel.AccelerateCallback

加速手段：
- 启用 TF32（Ampere+ GPU，MLP 矩阵乘提速约 5%，精度损失对 RL 可忽略）；
- 可选 torch.compile（默认关闭：编译后的 OptimizedModule 在
  model.save/load 时存在 state_dict 前缀不一致风险，且实测收益 <5%）。
"""

from __future__ import annotations

import torch
from stable_baselines3.common.callbacks import BaseCallback


class AccelerateCallback(BaseCallback):
    """在 model.learn() 开始时一次性配置加速选项。

    :param compile_policy: 是否对 actor/critic 做 torch.compile（默认 False）。
    """

    def __init__(self, compile_policy: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.compile_policy = compile_policy

    def _init_callback(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if self.verbose:
            print("[AccelerateCallback] TF32 enabled")

        if self.compile_policy:
            policy = self.model.policy
            policy.actor = torch.compile(policy.actor, mode="reduce-overhead")
            policy.critic = torch.compile(policy.critic, mode="reduce-overhead")
            if self.verbose:
                print("[AccelerateCallback] torch.compile(reduce-overhead) applied")

    def _on_step(self) -> bool:
        return True
