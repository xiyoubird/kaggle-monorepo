"""M4 · 系统性扫描"哪些分组键藏着信号"。

## 为什么值得做这个扫描

M3 里我们偶然发现：精确的 Flight Distance 做目标编码能涨 0.0014，
而"距离分箱 × 舱位 × 出行目的"却毫无增益。这说明**信号藏在某些特定的
高基数结构里**，而靠人工猜想很难猜中。

与其一个个人肉试，不如把它们全测一遍：

  - 单列：21 个原始列，每列做一次目标编码，看单特征 AUC；
  - 两列组合：21 选 2 共 210 对，同样各测一次。

每测一个键只需要做一次 groupby，**不需要训练模型**，所以整个扫描
几分钟就能跑完——比"训练 200 个模型"便宜好几个数量级。

## 怎么读结果

单特征 AUC 高只说明"这个键有信号"，不说明"加进模型有增量"——
因为树模型可能已经能从原始列里学到它（这正是 M2 特征工程失败的原因）。
所以扫描的作用是**圈定候选**，之后再对候选做真实的增量实验。

另外要小心高基数陷阱：如果一个键几乎每行都不同（比如
距离×年龄），它的目标编码就是在背答案，训练集上看起来很强，
验证集上等于噪声。所以这里限定中位数组大小不小于 50。

用法：
    python src/09_key_scan.py            # 单列 + 两列组合
    python src/09_key_scan.py --singles  # 只扫单列
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import OUT_DIR, SEED, encode_target, load_data
from target_encoding import apply_encoding, fit_encoding
from sklearn.model_selection import StratifiedKFold

MIN_MEDIAN_GROUP = 50  # 中位数组大小的下限，过滤"约等于背答案"的键


def oof_encode(keys: np.ndarray, y: np.ndarray, smooth: float, n_splits: int = 5):
    """对给定分组键做折内目标编码，返回 OOF 编码值数组。"""
    keys = pd.Series(keys)
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, va in skf.split(keys, y):
        table = fit_encoding(keys.iloc[tr], y[tr], float(y[tr].mean()), smooth)
        oof[va] = apply_encoding(keys.iloc[va], table, float(y[tr].mean()))
    return oof


def key_stats(keys: np.ndarray) -> tuple[int, float]:
    """返回 (不同取值个数, 中位数组大小)，用于判断这个键是否退化。"""
    vc = pd.Series(keys).value_counts()
    return len(vc), float(vc.median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", action="store_true", help="只扫单列")
    ap.add_argument("--smooth", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    train, _test, _sample, target = load_data()
    y = encode_target(train[target])
    base_cols = [c for c in train.columns if c not in (target, "id")]

    # 统一转成字符串，方便拼接；数值列保留原始取值不取整，
    # 因为 M3 已经证明"精确值"比"分箱值"更有信号。
    as_str = {c: train[c].astype(str).to_numpy() for c in base_cols}

    cands: list[tuple[str, np.ndarray]] = [(c, as_str[c]) for c in base_cols]
    if not args.singles:
        for a, b in itertools.combinations(base_cols, 2):
            cands.append((f"{a} × {b}", np.char.add(np.char.add(as_str[a], "|"), as_str[b])))

    print(f"待扫描的键：{len(cands)} 个（平滑 m={args.smooth}）\n")
    rows = []
    t0 = time.time()
    for i, (name, keys) in enumerate(cands, 1):
        n_unique, median_size = key_stats(keys)
        if median_size < MIN_MEDIAN_GROUP:
            continue  # 退化成近乎唯一的键，编码没有泛化意义
        enc = oof_encode(keys, y, args.smooth)
        auc = roc_auc_score(y, enc)
        rows.append({
            "key": name, "n_unique": n_unique,
            "median_group": round(median_size, 1), "auc": auc,
        })
        if i % 40 == 0:
            print(f"  ...已扫描 {i}/{len(cands)}，用时 {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(f"\n扫描完成，有效键 {len(df)} 个，用时 {time.time() - t0:.0f}s\n")
    print("单键目标编码的独立 AUC 排名（前 %d）：" % args.top)
    print(df.head(args.top).to_string(index=False))

    out = OUT_DIR / "key_scan.csv"
    df.to_csv(out, index=False)
    print(f"\n[记录] {out}")


if __name__ == "__main__":
    main()
