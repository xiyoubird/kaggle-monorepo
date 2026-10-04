"""公共工具：数据加载、交叉验证划分、提交文件写出。

把这三件事抽出来，是因为所有实验脚本都要用它们。
统一之后才能保证"不同模型用的是同一套折"，这样它们的 OOF 分数才可比较、
才能安全地融合。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 固定的随机种子：让实验可复现，也让融合时各模型共享同一套折
SEED = 42
N_SPLITS = 5

# 二分类正类的可能写法，用于自动识别目标列
_POSITIVE_LABELS = {"satisfied", "yes", "1", "true", "1.0"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """读取 train / test / sample_submission，并推断目标列名。"""
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    # 目标列 = 在 train 里有、在 test 里没有、且不是 id 的那一列
    target = next(
        c for c in train.columns
        if c not in test.columns and c.lower() != "id"
    )
    return train, test, sample, target


def encode_target(y: pd.Series) -> np.ndarray:
    """把目标转成 0/1。AUC 只关心顺序，所以约定"我们关注的那一类"为 1。

    对本题（Airline Satisfaction）我们关心的是"satisfied"，因此把
    satisfied 映射为 1。如果是字符串标签，按语义匹配；否则用第二类作正类。
    """
    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.unique(y.dropna()))
        if set(uniq) <= {0, 1}:
            return y.astype(int).to_numpy()
        # 非 0/1 的数值标签：把较大的值当作正类
        return (y > np.median(y)).astype(int).to_numpy()

    labels = y.astype(str).str.strip().str.lower()
    if set(labels.unique()) <= _POSITIVE_LABELS | {"neutral or dissatisfied", "no", "0", "false"}:
        return labels.isin(_POSITIVE_LABELS).astype(int).to_numpy()

    # 兜底：把出现次数较少的那一类当正类（罕见类通常是我们想找的）
    minority = labels.value_counts().idxmin()
    return (labels == minority).astype(int).to_numpy()


def split_features(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """切出特征矩阵，并返回类别列名单。

    做法：
    - 丢掉 `id` 与目标列；
    - 把"唯一值不超过 50 个"的非数值列统一声明为 pandas category，
      这样 LightGBM / CatBoost / XGBoost 都能按类别处理，而不是
      错误地当成连续数字（例如把 "Business" 当成 2 > "Economy" = 1）。
    """
    drop = [c for c in (target, "id") if c in train.columns]
    X = train.drop(columns=drop).copy()
    X_test = test.drop(columns=[c for c in drop if c in test.columns]).copy()

    # test 可能没有 id 列以外的差异，这里保证列顺序一致
    X_test = X_test[X.columns]

    # 非数值列一律当类别处理
    cat_cols = [c for c in X.columns if (not pd.api.types.is_numeric_dtype(X[c]))]

    for c in cat_cols:
        # 用所有取值（train+test）建立统一的类别字典，避免两边编码不一致
        categories = pd.Index(sorted(set(X[c].dropna().astype(str)) | set(X_test[c].dropna().astype(str))))
        X[c] = pd.Categorical(X[c].astype(str), categories=categories)
        X_test[c] = pd.Categorical(X_test[c].astype(str), categories=categories)

    return X, X_test, cat_cols


def make_folds(y: np.ndarray):
    """分层 K 折。

    用 StratifiedKFold 而不是普通 KFold：保证每折里正负样本比例一致，
    否则各折 AUC 会因为比例波动而互相不可比。
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def write_submission(test: pd.DataFrame, preds: np.ndarray, path: Path | None = None) -> Path:
    """按 sample_submission 的列结构写出提交文件。"""
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    id_col = sample.columns[0]
    pred_col = sample.columns[1]

    sub = pd.DataFrame({id_col: test[id_col].to_numpy(), pred_col: preds})
    path = path or (OUT_DIR / "submission.csv")
    sub.to_csv(path, index=False)
    print(f"[提交文件] {path}  ({len(sub):,} 行)")
    return path


def print_cv_summary(name: str, fold_scores: list[float], oof_auc: float) -> None:
    print(f"\n===== {name} =====")
    for i, s in enumerate(fold_scores, 1):
        print(f"  fold {i}: {s:.6f}")
    print(f"  平均折 AUC : {np.mean(fold_scores):.6f}  (±{np.std(fold_scores):.6f})")
    print(f"  OOF 总体 AUC: {oof_auc:.6f}")
