"""M4 · 模型对比与保存预测，为融合做准备。

## 为什么值得换模型

三个梯度提升库的内部机制不同：
  - LightGBM：直方图分裂 + 叶子优先生长，快；
  - XGBoost：精确贪心 + 深度优先，正则更严格；
  - CatBoost：对称树 + 有序目标统计，对小类别特别稳。

机制不同 → 犯错的方式不同 → 融合时能互补。
这是"融合"能涨分的**唯一**理由：如果两个模型错得一模一样，平均它们等于什么都没做。

## 为什么先固定表示

这一轮只换模型，特征表示固定为目前最优的"精确距离目标编码（m=5）"。
否则同时改动模型和特征，分数变化就说不清是谁的功劳了。

用法：
    python src/06_models.py lgb xgb cat
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR, SEED, encode_target, load_data, split_features
from cv_runner import run_cv_te


def key_distance(df: pd.DataFrame) -> pd.Series:
    """分组键：精确航线距离。这是 M3 里唯一被验证有效的表示。"""
    return df["Flight Distance"].astype(str)


def family_of(name: str) -> str:
    """把 "lgb_et"/"xgb_deep" 这类实验名映射回底层库。

    run_cv_te 只认识 lgb / xgb / cat 三种实现，
    但实验名需要带上变体后缀以便区分保存的预测文件。
    """
    return name.split("_")[0]


MODELS = {
    "lgb": {
        "params": {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 64, "min_child_samples": 50,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l1": 0.1, "lambda_l2": 0.1,
            "n_jobs": -1, "verbosity": -1, "seed": SEED,
        },
        "desc": "LightGBM（直方图分裂）",
    },
    "xgb": {
        "params": {
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist", "learning_rate": 0.03, "max_depth": 7,
            "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 1.0, "reg_alpha": 0.1,
            "nthread": -1, "seed": SEED,
        },
        "desc": "XGBoost（深度优先 + 强正则）",
    },
    "cat": {
        "params": {
            # 注：最初设成 depth=7/iterations=5000，单折要跑十几分钟，
            # 5 折远超可接受范围。CatBoost 的对称树本身很深时就慢，
            # 而本数据集信噪比高、不需要那么深的树，所以降到 6/3000。
            "iterations": 3000, "learning_rate": 0.08, "depth": 6,
            "l2_leaf_reg": 3.0, "eval_metric": "AUC",
            "loss_function": "Logloss", "random_seed": SEED,
            "od_type": "Iter", "od_wait": 100,
            "thread_count": -1, "allow_writing_files": False,
        },
        "desc": "CatBoost（对称树 + 有序目标统计）",
    },
    # ---- 下面几个不是"更强的模型"，而是"结构不同、犯错方式不同"的模型 ----
    # 上一轮融合的结论是：贪心选择把最大权重给了相关性最低（0.97）的弱模型，
    # 而不是最强的模型。所以这里刻意制造多样性。
    "lgb_et": {
        "params": {
            # extra_trees=True 让分裂阈值随机化，树的形状和标准 GBDT 明显不同
            "objective": "binary", "metric": "auc", "extra_trees": True,
            "learning_rate": 0.04, "num_leaves": 80, "min_child_samples": 60,
            "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
            "lambda_l1": 0.2, "lambda_l2": 1.0,
            "n_jobs": -1, "verbosity": -1, "seed": SEED + 7,
        },
        "desc": "LightGBM + extra_trees（随机化分裂阈值）",
    },
    "lgb_goss": {
        "params": {
            # GOSS 只对梯度大的样本做分裂，采样方式和标准 GBDT 不同
            "objective": "binary", "metric": "auc", "boosting": "goss",
            "learning_rate": 0.05, "num_leaves": 48, "min_child_samples": 40,
            "feature_fraction": 0.7, "lambda_l1": 0.1, "lambda_l2": 0.5,
            "n_jobs": -1, "verbosity": -1, "seed": SEED + 13,
        },
        "desc": "LightGBM + GOSS（单边梯度采样）",
    },
    "xgb_deep": {
        "params": {
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist", "grow_policy": "lossguide",
            "learning_rate": 0.05, "max_depth": 10, "max_leaves": 96,
            "min_child_weight": 40, "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_lambda": 3.0, "reg_alpha": 0.5,
            "nthread": -1, "seed": SEED + 21,
        },
        "desc": "XGBoost + lossguide 深树",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help=f"可选：{list(MODELS)}")
    ap.add_argument("--smooth", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lgb-params", default=None,
                    help="从 08_optuna.py 产出的 JSON 读取 LightGBM 最优超参")
    args = ap.parse_args()

    names = args.models or list(MODELS)
    for n in names:
        if n not in MODELS:
            raise SystemExit(f"未知模型：{n}，可选 {list(MODELS)}")

    train, test, _sample, target = load_data()
    y = encode_target(train[target])
    X, X_test, _X_ext, cat_cols = split_features(train, test, target)
    specs = [("te_dist", key_distance(train).to_numpy(), key_distance(test).to_numpy())]

    if args.lgb_params:
        blob = json.loads(Path(args.lgb_params).read_text())
        best = dict(blob["best_params"])
        # 注意：smooth 是"目标编码的平滑强度"，不是 LightGBM 的参数，
        # 必须先从字典里摘出来，否则会被当成未知参数传给 LightGBM。
        # （第一版忘了这步，虽然 LightGBM 只是忽略并警告，但语义是错的。）
        tuned_smooth = best.pop("smooth", None)
        MODELS["lgb"]["params"] = {
            **MODELS["lgb"]["params"], **best,
            # 这两个是 LightGBM 的运行环境参数，不属于被搜索的对象
            "objective": "binary", "metric": "auc",
            "n_jobs": -1, "verbosity": -1, "seed": SEED,
        }
        if tuned_smooth is not None and args.smooth == 5.0:
            args.smooth = float(tuned_smooth)
        print(f"[超参] 已载入 {args.lgb_params}，smooth={args.smooth}")
        print(json.dumps(best, indent=2, ensure_ascii=False))

    rows = []
    for name in names:
        cfg = MODELS[name]
        run_name = f"{name}{args.tag}"
        print(f"\n{'=' * 72}\n[{run_name}] {cfg['desc']}  平滑 m={args.smooth}")
        t0 = time.time()
        res = run_cv_te(
            family_of(name), X, y, X_test, cfg["params"], cat_cols, specs,
            smooth=args.smooth,
        )
        dt = time.time() - t0
        print(f"  >> OOF AUC = {res['auc']:.6f}  (耗时 {dt:.0f}s, 平均迭代 {int(np.mean(res['best_iters']))})")

        np.save(OUT_DIR / f"oof_{run_name}.npy", res["oof"])
        np.save(OUT_DIR / f"test_{run_name}.npy", res["test_pred"])
        rows.append({
            "name": run_name, "desc": cfg["desc"], "auc": res["auc"],
            "n_features": X.shape[1], "ext_weight": 0.0, "seconds": round(dt, 1),
        })

    log_path = OUT_DIR / "experiments.csv"
    out = pd.DataFrame(rows)
    if log_path.exists():
        old = pd.read_csv(log_path)
        out = pd.concat([old[~old["name"].isin(out["name"])], out], ignore_index=True)
    out.sort_values("auc", ascending=False).to_csv(log_path, index=False)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
