"""M4 · 超参数搜索（Optuna）。

## 为什么手动调参不够

梯度提升有 6-8 个相互影响的超参数。手工一次改一个，既慢，
又会因为"改 A 时用着旧的 B"而错过 A、B 需要同时变化的区域。
Optuna 的 TPE 采样器会根据历史结果把搜索集中到有希望的区域，
比网格搜索高效得多。

## 两个刻意的设计取舍

1. **用 3 折而不是 5 折搜索**：每组参数跑 5 折约 110 秒，
   跑 40 组要 70 多分钟。3 折约 60 秒，能多试近一倍参数。
   代价是 3 折的 AUC 绝对值偏低、噪声稍大——但它衡量的是
   **参数的相对优劣**，这个排序在 3 折下通常是稳的。
   最后我会用 5 折重新验证选出来的参数。

2. **搜索空间包含"更小更正则"的方向**：M3 的结论是
   "更大容量没有帮助"，所以与其继续往大调，不如让搜索同时
   探索学习率、叶子数、采样比例和正则强度的组合。

用法：
    python src/08_optuna.py --trials 40 --n-splits 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from common import OUT_DIR, SEED, encode_target, load_data, split_features
from cv_runner import run_cv_te
from target_encoding import build_keys

# Optuna 默认会为每次 trial 打印几行日志，上万行输出会淹没真正的结果
optuna.logging.set_verbosity(optuna.logging.WARNING)


def key_distance(df):
    return df["Flight Distance"].astype(str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=None,
                    help="固定平滑强度；不指定则由 Optuna 在 [1,20] 内搜索")
    ap.add_argument("--timeout", type=int, default=None, help="搜索总时限（秒）")
    ap.add_argument("--study", default="lgb_te")
    args = ap.parse_args()

    train, test, _sample, target = load_data()
    y = encode_target(train[target])
    X, X_test, _X_ext, cat_cols = split_features(train, test, target)
    specs = [("te_dist", key_distance(train).to_numpy(), key_distance(test).to_numpy())]

    def objective(trial: optuna.Trial) -> float:
        smooth = args.smooth
        if smooth is None:
            # 把平滑强度也交给 Optuna：它和目标编码这个特征本身是否好用直接相关
            smooth = trial.suggest_float("smooth", 1.0, 20.0)
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 20.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "n_jobs": -1, "verbosity": -1, "seed": SEED,
        }
        res = run_cv_te(
            "lgb", X, y, X_test, params, cat_cols, specs,
            smooth=smooth, n_splits=args.n_splits, verbose=False,
        )
        trial.set_user_attr("mean_best_iter", float(np.mean(res["best_iters"])))
        return res["auc"]

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=args.study)

    t0 = time.time()

    def callback(st, tr):
        print(f"  trial {tr.number:3d}  AUC={tr.value:.6f}  "
              f"最优={st.best_value:.6f}  用时 {time.time() - t0:6.0f}s")

    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, callbacks=[callback],
    )

    print(f"\n{'=' * 72}")
    print(f"搜索完成：{len(study.trials)} 组，最好 {args.n_splits} 折 AUC = {study.best_value:.6f}")
    print("最优参数：")
    print(json.dumps(study.best_params, indent=2, ensure_ascii=False))

    out = OUT_DIR / f"optuna_{args.study}_best.json"
    out.write_text(json.dumps({
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_splits": args.n_splits,
        "smooth": args.smooth,
        "n_trials": len(study.trials),
    }, indent=2, ensure_ascii=False))
    print(f"[记录] {out}")


if __name__ == "__main__":
    main()
