"""可复用的交叉验证运行器。

为什么单独抽一个模块：
  进入 M3 之后要做几十组对比实验（不同特征、不同模型、不同权重）。
  如果每次实验都复制一遍训练循环，很快就会"两组实验用了不同的折/不同的早停规则"，
  结论也就不再可比较。这里把规则**固定下来**：

    - 折的划分永远来自 common.make_folds（同一个随机种子）；
    - OOF 永远只在我们自己的比赛训练集上计算；
    - 早停永远只看我们自己的验证折（即使训练时混入了外部数据）。

  最后一条尤其重要：外部数据的分布和比赛数据不完全一样，
  如果拿外部数据来做早停，选出的迭代次数是"对原始数据最好"而非"对比赛最好"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import make_folds


def run_lgb_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    cat_cols: list[str],
    X_ext: pd.DataFrame | None = None,
    y_ext: np.ndarray | None = None,
    ext_weight: float = 1.0,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    verbose: bool = True,
) -> dict:
    """跑一次 5 折 LightGBM，返回 OOF / 测试集预测 / 分数。"""
    import lightgbm as lgb

    folds = make_folds(y)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        w_tr = None

        if X_ext is not None and ext_weight > 0:
            # 外部数据全部加入每一折的训练集（它不是验证集的一部分）
            X_tr = pd.concat([X_tr, X_ext], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, y_ext])
            # 降权：让外部数据对损失函数的贡献小于比赛数据
            w_tr = np.concatenate([
                np.ones(len(tr_idx), dtype=np.float64),
                np.full(len(X_ext), float(ext_weight), dtype=np.float64),
            ])

        dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            params, dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)
        fold_scores.append(roc_auc_score(y[va_idx], oof[va_idx]))
        best_iters.append(model.best_iteration)
        if verbose:
            print(f"    fold {i}: AUC={fold_scores[-1]:.6f}  best_iter={model.best_iteration}")

    auc = roc_auc_score(y, oof)
    return {
        "oof": oof,
        "test_pred": test_pred,
        "auc": auc,
        "fold_scores": fold_scores,
        "best_iters": best_iters,
    }


def run_xgb_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    X_ext: pd.DataFrame | None = None,
    y_ext: np.ndarray | None = None,
    ext_weight: float = 1.0,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    verbose: bool = True,
) -> dict:
    """同样规则的 XGBoost 版本，保证与 LightGBM 的折完全一致、可直接融合。"""
    import xgboost as xgb

    folds = make_folds(y)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        w_tr = None
        if X_ext is not None and ext_weight > 0:
            X_tr = pd.concat([X_tr, X_ext], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, y_ext])
            w_tr = np.concatenate([
                np.ones(len(tr_idx)), np.full(len(X_ext), float(ext_weight)),
            ])

        dtr = xgb.DMatrix(X_tr, y_tr, weight=w_tr, enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], y[va_idx], enable_categorical=True)

        model = xgb.train(
            params, dtr,
            num_boost_round=num_boost_round,
            evals=[(dva, "valid")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )

        oof[va_idx] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / len(folds)
        fold_scores.append(roc_auc_score(y[va_idx], oof[va_idx]))
        best_iters.append(model.best_iteration + 1)
        if verbose:
            print(f"    fold {i}: AUC={fold_scores[-1]:.6f}  best_iter={best_iters[-1]}")

    auc = roc_auc_score(y, oof)
    return {
        "oof": oof, "test_pred": test_pred, "auc": auc,
        "fold_scores": fold_scores, "best_iters": best_iters,
    }
