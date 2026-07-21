（自定义）=

# 自定义环境

添加对自定义环境的支持的最简单方法是编辑
`rl_zoo3/import_envs.py` 并在此处注册您的环境。然后你
需要在超参数文件中为其添加一个部分
（`hyperparams/algo.yml` 或您可以指定的自定义 yaml 文件
使用 `--conf-file` 参数）。
