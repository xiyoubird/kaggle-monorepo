"""M1 · 探索性数据分析 (EDA)。

思路：
在建模之前必须回答四个问题，否则后面所有实验都是盲猜：
  1. 数据有多大？训练/测试各多少行？
  2. 目标变量是什么、正样本占比多少？（决定是否要关心不平衡）
  3. 有哪些列、什么类型？（决定特征工程方向）
  4. 缺失值在哪里、有多少？（决定是否需要填补，以及"缺失"本身是否是信号）

用法：
    python src/01_eda.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def profile(name: str, df: pd.DataFrame, target: str | None) -> None:
    print("=" * 78)
    print(f"{name}: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print("=" * 78)

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(info.to_string())

    # 数值列的分布概览
    num = df.select_dtypes("number")
    if not num.empty:
        print("\n-- 数值列统计 --")
        print(num.describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())

    # 类别列（唯一值较少的）列出取值
    for col in df.columns:
        if col == target:
            continue
        n = df[col].nunique(dropna=False)
        if n <= 12:
            print(f"\n-- {col} 取值分布 --")
            print(df[col].value_counts(dropna=False).to_string())

    if target and target in df.columns:
        print(f"\n-- 目标变量 {target} --")
        vc = df[target].value_counts(dropna=False)
        print(vc.to_string())
        print("正样本占比:", round(vc.iloc[0] / len(df), 4))


def main() -> None:
    files = sorted(p for p in RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"没找到 CSV，请先运行 00_download_data.py。目录：{RAW_DIR}")

    # 先自动找目标列名：train 里有、test 里没有的那一列
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"

    if train_path.exists() and test_path.exists():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        target = next((c for c in train.columns if c not in test.columns and c != "id"), None)
        print(f"推断目标列 = {target}\n")
        profile("train.csv", train, target)
        profile("test.csv", test, None)
    else:
        # 兜底：把所有 csv 都打印一遍，方便人工判断
        for f in files:
            df = pd.read_csv(f)
            profile(f.name, df, None)

    for extra in ("sample_submission.csv",):
        p = RAW_DIR / extra
        if p.exists():
            print("\n" + "=" * 78)
            print(f"{extra} 前 5 行")
            print("=" * 78)
            print(pd.read_csv(p).head().to_string())


if __name__ == "__main__":
    main()
