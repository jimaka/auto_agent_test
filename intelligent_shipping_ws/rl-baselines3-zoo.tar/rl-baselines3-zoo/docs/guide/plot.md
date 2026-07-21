（情节）=

# 情节脚本

绘图脚本（有待记录，请参阅 SB3 中的“结果”部分
文档）：

- `scripts/all_plots.py`/`scripts/plot_from_file.py` 用于绘图评估
- `scripts/plot_train.py` 用于绘制训练奖励/成功

## 示例

绘制训练成功率（y 轴）w.r.t.时间步长（x 轴）与移动
包含 `HER` 的所有 `Fetch` 环境的 500 集窗口
算法：

```
python scripts/plot_train.py -a her -e Fetch -y success -f rl-trained-agents/ -w 500 -x steps
```

在 HalfCheetah 上绘制 TQC、SAC 和 TD3 的评估奖励曲线
Ant PyBullet 环境：

```
python3 scripts/all_plots.py -a sac td3 tqc --env HalfCheetahBullet AntBullet -f rl-trained-agents/
```

## 使用 rliable 库进行绘图

RL 动物园整合了一些
[rliable](https://agarwl.github.io/rliable/) 库功能。你
可以在这个[博客中找到 rliable 使用的工具的直观解释
帖子](https://araffin.github.io/post/rliable/)。

首先，您需要安装
[可靠](https://github.com/google-research/rliable)。

注意：在这种情况下需要 Python 3.7+。

然后使用 `all_plots.py` 脚本将结果导出到文件
（见上文）：

```
python scripts/all_plots.py -a sac td3 tqc --env Half Ant -f logs/ -o logs/offpolicy
```

您现在可以将 `plot_from_file.py` 脚本与 `--rliable` 一起使用，
`--versus` 和 `--iqm` 参数：

```
python scripts/plot_from_file.py -i logs/offpolicy.pkl --skip-timesteps --rliable --versus -l SAC TD3 TQC
```

：：：{笔记}
您可能需要编辑 `plot_from_file.py`，特别是
`env_key_to_env_id` 字典和
`scripts/score_normalization.py` 存储最小和最大分数
每个环境。
:::

备注：使用 `--rliable` 选项绘图通常很慢，因为
置信区间需要使用引导抽样来计算。
