# kaggle-monorepo

个人 Kaggle 竞赛 monorepo。每个竞赛一个目录，互不干扰。

## 目录约定

```
competitions/<competition-slug>/
├── README.md          # 竞赛背景 + 我的方案 + 实验日志
├── docs/              # 逐步思路记录
├── src/               # 可复现的脚本
├── configs/           # 超参数 / 实验配置
├── data/              # 原始与处理后的数据（git 忽略）
└── outputs/           # 提交文件与日志（git 忽略）
```

## 当前竞赛

| 竞赛 | 指标 | 状态 |
|---|---|---|
| [playground-series-s6e10](competitions/playground-series-s6e10/README.md) | ROC AUC | 进行中 |
