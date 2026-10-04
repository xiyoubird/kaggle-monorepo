# 01 · 环境与凭据

## 需要什么

1. **Kaggle API 凭据**：下载比赛数据和提交结果都需要它。
   形式是一个 `kaggle.json`，内容形如：

   ```json
   {"username": "你的用户名", "key": "一串密钥"}
   ```

   获取方式：Kaggle 网站 → 右上角头像 → **Settings** → **API** →
   **Create New Token**，会下载一个 `kaggle.json`。
   把它放到 `~/.kaggle/kaggle.json`（权限 600）即可。

2. **Python 环境**：用 `~/anaconda3` 的 Python 3.11。
   已有 pandas / numpy / scikit-learn，需要补装：
   `kaggle`、`lightgbm`、`xgboost`、`catboost`、`optuna`。

## 为什么用 LightGBM / XGBoost / CatBoost 而不是深度学习

这是**表格数据二分类**问题，样本量在几十万级别。在这个规模上：

- 梯度提升树（GBDT）是公认的强基线，训练快、几乎不用调就能打得很好；
- 神经网络对表格数据通常需要大量调参才能打平 GBDT，性价比低；
- 三个不同的 GBDT 实现内部机制不同（直方图分裂 / 精确贪心 / 对称树），
  融合时**误差模式不同**，能带来真实的增益。

所以策略是：三个 GBDT + 融合，而不是一上来就上深度学习。
