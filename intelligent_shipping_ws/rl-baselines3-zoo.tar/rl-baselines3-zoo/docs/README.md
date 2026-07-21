## RL Zoo3 文档

此文件夹包含 RL Zoo 的文档。


### 构建文档

#### 安装 Sphinx 和主题
在项目根目录执行此命令：
```
pip install stable_baselines3[docs]
pip install -e .
```

#### 构建文档

在`docs/`文件夹中：
```
make html
```

如果您想在每次更改文件时进行构建：

```
sphinx-autobuild . _build/html
```
