"""M2 · 单模型基线：LightGBM + 5 折交叉验证。

思路（为什么先做这个）：
  1. 先要一个"能跑完、能得分"的东西，验证整条链路（数据→训练→提交）没坏。
  2. LightGBM 在表格数据上几乎不需要精细调参就能给出强结果，
     适合作为后续所有改进的参照物（benchmark）。
  3. 用早停 (early stopping)：让模型自己在验证集上决定训练多久，
     比起手工设定迭代次数更稳，也不容易过拟合。

用法：
    python src/02_baseline_lgbm.py
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common import (
    OUT_DIR, SEED, encode_target, load_data, make_folds,
    print_cv_summary, split_features, write_submission,
)


def main() -> None:
    train, test, _sample, target = load_data()
    print(f"训练集 {train.shape} / 测试集 {test.shape} / 目标列 '{target}'")

    y = encode_target(train[target])
    X, X_test, _X_ext, cat_cols = split_features(train, test, target)
    print(f"特征数 {X.shape[1]}，其中类别列 {len(cat_cols)} 个：{cat_cols}")
    print(f"正样本占比 {y.mean():.4f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "n_jobs": -1,
        "verbosity": -1,
        "seed": SEED,
    }

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    folds = make_folds(y)
    fold_scores: list[float] = []

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            params, dtr,
            num_boost_round=5000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )

        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        # 每折对测试集的预测取平均，是降低方差的标准做法
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)

        auc = roc_auc_score(y[va_idx], oof[va_idx])
        fold_scores.append(auc)
        print(f"  fold {i}: AUC={auc:.6f}  best_iter={model.best_iteration}")

    oof_auc = roc_auc_score(y, oof)
    print_cv_summary("LightGBM baseline", fold_scores, oof_auc)

    np.save(OUT_DIR / "oof_lgbm.npy", oof)
    np.save(OUT_DIR / "test_lgbm.npy", test_pred)
    write_submission(test, test_pred, OUT_DIR / "submission_lgbm.csv")


if __name__ == "__main__":
    main()
