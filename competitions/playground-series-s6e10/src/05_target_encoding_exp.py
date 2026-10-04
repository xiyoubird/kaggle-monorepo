"""M3 · 目标编码实验。

## 背景（第一轮实验的结论）

题目里没有"航线"这一列。第一轮实验发现：

  - 用 `Class × 出行目的 × 距离分箱(每100英里)` 做编码：**无增益**（0.958962）
  - 用**精确的 Flight Distance** 做编码：**+0.0014**（0.960323）

这个对比本身很有信息量——说明信号存在于"每一条具体航线"里，
而不是"哪一类航线"里。分箱把 3474 条航线粗暴地合并成几十组，
信号就被平均掉了。

## 这一轮要回答

  Q1 平滑强度 m 取多少最好？（m 太小编码是噪声，太大会被拉回全局均值）
  Q2 更细的交互键（精确距离 × 舱位 × 出行目的）能否超过单用距离？
  Q3 目标编码能否和之前的特征工程叠加？

用法：
    python src/05_target_encoding_exp.py dist all --smooth 20
    python src/05_target_encoding_exp.py dist --smooth 5 --tag _m5
"""

from __future__ import annotations

import argparse
import time
from importlib import import_module

import numpy as np
import pandas as pd

from common import OUT_DIR, encode_target, load_data, split_features
from cv_runner import run_lgb_cv_te
from features import add_features
from target_encoding import build_keys

PARAMS_BASE = import_module("04_experiments").PARAMS_BASE


# ---------------- 分组键的定义 ----------------

def k_dist(df: pd.DataFrame) -> pd.Series:
    """精确航线距离。目前最强的单键。"""
    return df["Flight Distance"].astype(str)


def k_fine(df: pd.DataFrame) -> pd.Series:
    """精确距离 × 舱位 × 出行目的。

    比 k_dist 多两个维度，但每组样本数大约只剩 1/6，平滑会把弱组拉回全局均值。
    所以它不一定更好——必须用 CV 判断。
    """
    return build_keys(df, ["Flight Distance", "Class", "Type of Travel"])


def k_route_bin(df: pd.DataFrame) -> pd.Series:
    """粗分箱航线画像，作为对照（已知无增益）。"""
    return build_keys(
        df.assign(_b=(df["Flight Distance"] // 100).astype(str)),
        ["Class", "Type of Travel", "_b"],
    )


def k_age(df: pd.DataFrame) -> pd.Series:
    return df["Age"].astype(str)


def k_cat(df: pd.DataFrame) -> pd.Series:
    return build_keys(df, ["Gender", "Customer Type", "Type of Travel", "Class"])


KEY_FUNCS = {
    "dist": k_dist,
    "fine": k_fine,
    "route": k_route_bin,
    "age": k_age,
    "cat": k_cat,
}

# 每个配置 = 用哪几个键 + 是否叠加特征工程
CONFIGS: dict[str, dict] = {
    "dist":     dict(keys=["dist"], fe=False),
    "fine":     dict(keys=["fine"], fe=False),
    "dist_age": dict(keys=["dist", "age"], fe=False),
    "all":      dict(keys=["dist", "fine", "age", "cat"], fe=False),
    "dist_fe":  dict(keys=["dist"], fe=True),
    "all_fe":   dict(keys=["dist", "fine", "age", "cat"], fe=True),
}


def log_result(name: str, desc: str, auc: float, n_features: int, seconds: float) -> None:
    row = pd.DataFrame([{
        "name": name, "desc": desc, "auc": auc, "n_features": n_features,
        "ext_weight": 0.0, "seconds": round(seconds, 1),
    }])
    log_path = OUT_DIR / "experiments.csv"
    if log_path.exists():
        old = pd.read_csv(log_path)
        row = pd.concat([old[~old["name"].isin(row["name"])], row], ignore_index=True)
    row.sort_values("auc", ascending=False).to_csv(log_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help=f"可选：{list(CONFIGS)}")
    ap.add_argument("--smooth", type=float, default=20.0, help="目标编码平滑强度 m")
    ap.add_argument("--tag", default="", help="结果文件名的附加标记")
    ap.add_argument("--no-fd", action="store_true", help="从特征中移除原始 Flight Distance 列")
    args = ap.parse_args()

    names = args.configs or list(CONFIGS)
    for n in names:
        if n not in CONFIGS:
            raise SystemExit(f"未知配置：{n}，可选 {list(CONFIGS)}")

    train, test, _sample, target = load_data()
    y = encode_target(train[target])

    for name in names:
        cfg = CONFIGS[name]
        feature_fn = add_features if cfg["fe"] else None

        # 分组键始终从完整数据构造（需要原始列）；
        # 是否把 Flight Distance 喂给模型由 --no-fd 决定。
        specs = [
            (k, KEY_FUNCS[k](train).to_numpy(), KEY_FUNCS[k](test).to_numpy())
            for k in cfg["keys"]
        ]

        train_feat = train.drop(columns=["Flight Distance"]) if args.no_fd else train
        test_feat = test.drop(columns=["Flight Distance"]) if args.no_fd else test
        X, X_test, _X_ext, cat_cols = split_features(
            train_feat, test_feat, target, feature_fn=feature_fn,
        )

        run_name = f"te_{name}{args.tag}" + ("_nofd" if args.no_fd else "")
        print(f"\n{'=' * 72}\n[{run_name}] 键={cfg['keys']} 特征工程={cfg['fe']} m={args.smooth}")

        t0 = time.time()
        res = run_lgb_cv_te(X, y, X_test, PARAMS_BASE, cat_cols, specs, smooth=args.smooth)
        dt = time.time() - t0
        print(f"  >> OOF AUC = {res['auc']:.6f}  (耗时 {dt:.0f}s)")

        np.save(OUT_DIR / f"oof_{run_name}.npy", res["oof"])
        np.save(OUT_DIR / f"test_{run_name}.npy", res["test_pred"])
        log_result(
            run_name,
            f"TE keys={cfg['keys']} fe={cfg['fe']} m={args.smooth}",
            res["auc"], X.shape[1], dt,
        )


if __name__ == "__main__":
    main()
