"""M3 · 对比实验：外部数据 / 特征工程 / 超参数。

这一阶段要回答三个互相独立的问题，所以我用"控制变量"的方式做：

  Q1 外部数据有用吗？该给多大权重？（固定特征与超参，只改 ext_weight）
  Q2 特征工程有用吗？（固定超参，只改 feature_fn）
  Q3 更强的超参有用吗？（固定数据，只改 params）

为什么先做权重扫描：外部数据来自原始分布，比赛数据来自合成分布，
两者并不完全一致。权重太大，模型会被带偏到原始分布上；
权重太小，外部数据的信息又用不上。这个最优值只能用 CV 找。

用法：
    python src/04_experiments.py                 # 跑全部
    python src/04_experiments.py ext0 ext05      # 只跑指定实验
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from common import (
    OUT_DIR, SEED, encode_target, load_data, load_external, split_features,
)
from cv_runner import run_lgb_cv
from features import add_features, add_features_minimal

# ---- 基础超参数：与 M2 基线一致，作为参照 ----
PARAMS_BASE = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 64, "min_child_samples": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "n_jobs": -1, "verbosity": -1, "seed": SEED,
}

# ---- 更强的一组超参数 ----
# 理由：基线用 num_leaves=64 时收敛很快（best_iter 约 700-950），
# 说明模型容量可能不够。合成数据信噪比高，通常吃得下更复杂的模型，
# 所以同时提高叶子数、降低学习率，并用更强的正则压住过拟合。
PARAMS_BIG = {
    **PARAMS_BASE,
    "learning_rate": 0.02,
    "num_leaves": 160,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
}


def build(feature_fn, use_external: bool, keep_id: bool = False):
    """按指定配置准备特征，返回 X, y, X_test, X_ext, y_ext, cat_cols, target, test。"""
    # 外部数据没有 id 列，两种设置不能同时开启
    assert not (keep_id and use_external), "keep_id 与 use_external 不能同时为真"
    train, test, _sample, target = load_data()
    y = encode_target(train[target])

    external = load_external(target) if use_external else None
    y_ext = encode_target(external[target]) if external is not None else None

    X, X_test, X_ext, cat_cols = split_features(
        train, test, target, external=external, feature_fn=feature_fn, keep_id=keep_id,
    )
    return X, y, X_test, X_ext, y_ext, cat_cols, target, test


EXPERIMENTS: dict[str, dict] = {
    # Q1：外部数据权重扫描（不加特征工程、用基础超参）
    "ext0":   dict(feature_fn=None, use_external=False, ext_weight=0.0,  params=PARAMS_BASE, desc="仅比赛数据（=M2 基线）"),
    "ext025": dict(feature_fn=None, use_external=True,  ext_weight=0.25, params=PARAMS_BASE, desc="+外部数据 w=0.25"),
    "ext05":  dict(feature_fn=None, use_external=True,  ext_weight=0.5,  params=PARAMS_BASE, desc="+外部数据 w=0.5"),
    "ext1":   dict(feature_fn=None, use_external=True,  ext_weight=1.0,  params=PARAMS_BASE, desc="+外部数据 w=1.0"),

    # Q2：特征工程（基础超参、仅比赛数据，隔离出特征的净效果）
    "fe_min":  dict(feature_fn=add_features_minimal, use_external=False, ext_weight=0.0, params=PARAMS_BASE, desc="+精简特征工程"),
    "fe_full": dict(feature_fn=add_features,         use_external=False, ext_weight=0.0, params=PARAMS_BASE, desc="+完整特征工程"),

    # Q3：超参数（仅比赛数据 + 完整特征）
    "fe_big": dict(feature_fn=add_features, use_external=False, ext_weight=0.0, params=PARAMS_BIG, desc="+完整特征 + 大容量超参"),

    # 组合：把上面各自的结论叠加起来
    "combo": dict(feature_fn=add_features, use_external=True, ext_weight=0.5, params=PARAMS_BIG, desc="完整特征 + 外部数据 w=0.5 + 大容量"),

    # 泄漏检验：把 id 本身当特征。
    # 合成数据有时是按 id 顺序分批生成的，导致 id 与目标存在虚假相关。
    # 如果这一项大幅提升，说明存在可利用的泄漏；但要注意它可能对私有榜无效。
    "id_leak": dict(feature_fn=None, use_external=False, ext_weight=0.0, params=PARAMS_BASE, desc="把 id 当特征（泄漏检验）", keep_id=True),
}


def main() -> None:
    names = sys.argv[1:] or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"未知实验名：{unknown}\n可选：{list(EXPERIMENTS)}")

    # 按 (特征工程, 是否用外部数据) 缓存数据，避免每个实验都重新读 CSV
    cache: dict[tuple, tuple] = {}
    rows = []

    for name in names:
        cfg = EXPERIMENTS[name]
        key = (cfg["feature_fn"], cfg["use_external"], cfg.get("keep_id", False))
        if key not in cache:
            cache[key] = build(cfg["feature_fn"], cfg["use_external"], cfg.get("keep_id", False))
        X, y, X_test, X_ext, y_ext, cat_cols, _target, _test = cache[key]

        print(f"\n{'=' * 72}\n[{name}] {cfg['desc']}")
        use_ext = X_ext is not None and cfg["ext_weight"] > 0
        print(f"  特征 {X.shape[1]} 列 | 训练 {len(X):,} 行"
              + (f" | 外部 {len(X_ext):,} 行 (权重 {cfg['ext_weight']})" if use_ext else ""))

        t0 = time.time()
        res = run_lgb_cv(
            X, y, X_test, cfg["params"], cat_cols,
            X_ext=X_ext if use_ext else None,
            y_ext=y_ext, ext_weight=cfg["ext_weight"],
        )
        dt = time.time() - t0
        print(f"  >> OOF AUC = {res['auc']:.6f}   (耗时 {dt:.0f}s, 平均迭代 {int(np.mean(res['best_iters']))})")

        # 保存每个实验的预测，方便之后做融合
        np.save(OUT_DIR / f"oof_{name}.npy", res["oof"])
        np.save(OUT_DIR / f"test_{name}.npy", res["test_pred"])

        rows.append({
            "name": name, "desc": cfg["desc"], "auc": res["auc"],
            "n_features": X.shape[1], "ext_weight": cfg["ext_weight"],
            "seconds": round(dt, 1),
        })

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print("\n\n" + "=" * 72)
    print("实验结果汇总（按 OOF AUC 降序）")
    print("=" * 72)
    print(df.to_string(index=False))

    log_path = OUT_DIR / "experiments.csv"
    if log_path.exists():
        old = pd.read_csv(log_path)
        df = pd.concat([old[~old["name"].isin(df["name"])], df], ignore_index=True)
    df.sort_values("auc", ascending=False).to_csv(log_path, index=False)
    print(f"\n[记录] {log_path}")


if __name__ == "__main__":
    main()
