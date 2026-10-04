"""M4 · 模型融合。

## 融合为什么有用

只有当各个模型**错得不一样**时，平均才有意义。
如果两个模型对所有样本的排序完全一致，融合等于在做无用功。
所以流程上先看单模型分数，再看它们的相关性，最后才决定融谁。

## 三种融合方式，从保守到激进

1. **等权平均**：不估任何参数，最稳。缺点是不能偏向更强的模型。
2. **排名平均**：先把预测值转成排名再平均。
   AUC 只看排序，所以排名空间里的平均通常比概率空间更契合指标
   （概率的绝对大小受各模型校准程度影响，排名不受影响）。
3. **优化权重**：在 OOF 上搜索一组使 AUC 最大的权重。
   风险是**在 OOF 上过拟合权重**——600 多个样本点上的搜索很容易
   找到"碰巧"好的权重，线上却没有这个优势。

## 怎么诚实地报告优化权重的分数

直接把"在全部 OOF 上优化出来的权重"的分数报出来是**自欺欺人**的，
因为那组权重见过这些样本。所以这里用嵌套交叉验证：
把 OOF 再切 5 折，每次在 4 折上优化权重、在剩下 1 折上评估。
得到的分数才是"这套权重搜索流程"的真实泛化能力。

用法：
    python src/07_blend.py                 # 自动纳入 outputs/oof_*.npy
    python src/07_blend.py lgb xgb cat     # 指定模型
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import OUT_DIR, encode_target, load_data, write_submission


def rank01(x: np.ndarray) -> np.ndarray:
    """转成 0-1 的排名，消除不同模型的校准差异。"""
    return rankdata(x) / len(x)


def fit_weights(R: np.ndarray, y: np.ndarray, n_steps: int = 30) -> np.ndarray:
    """Caruana 贪心集成选择：每次挑一个"加入后 AUC 提升最多"的模型。

    为什么不用 Nelder-Mead 直接优化权重：
      - 单纯形法在维度一高就会退化，而且每算一次 AUC 要 O(n log n)，
        十几维时单次搜索要十几分钟；
      - 贪心选择是**有放回**的（同一个模型可以被选多次），
        等价于在整数格点上搜索权重，天然非负、天然归一化，
        而且在 10-20 个模型这个规模上非常稳。

    这是 Caruana 等人在 "Ensemble Selection from Libraries of Models" 里
    提出的经典做法，至今仍是 Kaggle 上最常用的融合手段之一。
    """
    n = R.shape[1]
    counts = np.zeros(n)

    # 起点用单模型里最好的那个，避免第一步被弱模型带偏
    best_j = max(range(n), key=lambda k: roc_auc_score(y, R[:, k]))
    counts[best_j] = 1.0
    cur = R[:, best_j].copy()

    for _ in range(n_steps):
        total = counts.sum() + 1
        best_j, best_auc = None, -np.inf
        for j in range(n):
            a = roc_auc_score(y, (cur + R[:, j]) / total)
            if a > best_auc:
                best_auc, best_j = a, j
        counts[best_j] += 1
        cur += R[:, best_j]

    return counts / counts.sum()


def main() -> None:
    train, test, _sample, target = load_data()
    y = encode_target(train[target])

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = sorted(p.stem[4:] for p in OUT_DIR.glob("oof_*.npy"))

    oofs, tests, used = [], [], []
    for n in names:
        op, tp = OUT_DIR / f"oof_{n}.npy", OUT_DIR / f"test_{n}.npy"
        if not op.exists() or not tp.exists():
            print(f"  [跳过] {n}：缺少预测文件")
            continue
        o = np.load(op)
        if len(o) != len(y):
            print(f"  [跳过] {n}：OOF 长度 {len(o)} 与训练集 {len(y)} 不符")
            continue
        oofs.append(o)
        tests.append(np.load(tp))
        used.append(n)

    if not oofs:
        raise SystemExit("没有可用的预测文件")

    print("参与融合的模型：")
    singles = []
    for n, o in zip(used, oofs):
        a = roc_auc_score(y, o)
        singles.append(a)
        print(f"  {n:22s} OOF AUC = {a:.6f}")

    O = np.column_stack(oofs)
    T = np.column_stack(tests)
    RO = np.column_stack([rank01(o) for o in oofs])
    RT = np.column_stack([rank01(t) for t in tests])

    # 模型间相关性：太低说明有互补空间，太高说明融合收益有限
    print("\n模型预测的秩相关矩阵：")
    print(pd.DataFrame(np.corrcoef(RO.T), index=used, columns=used).round(4).to_string())

    print("\n融合结果：")
    results = {}
    results["等权概率平均"] = roc_auc_score(y, O.mean(axis=1))
    results["等权排名平均"] = roc_auc_score(y, RO.mean(axis=1))

    # 单模型里最好的那个，作为对照
    best_single = max(singles)
    results["单模型最好"] = best_single

    # 用嵌套交叉验证诚实评估"优化权重"这个流程
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    nested = np.zeros(len(y))
    for tr, va in skf.split(RO, y):
        w = fit_weights(RO[tr], y[tr])
        nested[va] = RO[va] @ w
    results["优化权重（嵌套CV）"] = roc_auc_score(y, nested)

    # 用全量 OOF 拟合最终权重（分数会偏乐观，仅用于生成提交文件）
    w_full = fit_weights(RO, y)
    results["优化权重（全量拟合，偏乐观）"] = roc_auc_score(y, RO @ w_full)

    for k, v in results.items():
        print(f"  {k:28s} {v:.6f}")

    print("\n最终权重：")
    for n, w in zip(used, w_full):
        print(f"  {n:22s} {w:.4f}")

    # 提交文件用排名平均形式写入（AUC 只看排序，绝对值不影响结果）
    final_test = RT @ w_full
    path = write_submission(test, final_test, OUT_DIR / "submission_blend.csv")
    print(f"[完成] 融合提交文件：{path}")

    np.save(OUT_DIR / "blend_oof.npy", RO @ w_full)
    np.save(OUT_DIR / "blend_test.npy", final_test)


if __name__ == "__main__":
    main()
