"""折内（out-of-fold）目标编码。

## 为什么必须放在折内

目标编码就是把"某个类别取值的平均目标值"当成特征。
如果直接在**全量训练集**上算好再拿去交叉验证，就会发生泄漏：
每一行的这个特征里都混进了它自己的目标值，验证分数会虚高，
但线上完全没有这个优势。所以必须：

    - 只用"当前折的训练部分"计算编码表；
    - 用这张表去转换"当前折的验证部分"和测试集。

这样验证折上的编码值完全来自它没见过的数据，才是诚实的估计。

## 为什么需要平滑

某个类别如果只出现 1 次且目标为 1，它的均值就是 1.0 —— 纯噪声。
平滑的写法是

    enc = (sum + prior * m) / (count + m)

出现次数少时结果被拉向全局均值 prior，出现次数多时才体现自己的均值。
`m` 就是"相当于掺入 m 个虚拟样本"。

## 为什么这在本题可能有用

`Flight Distance` 有 3474 个不同取值，可以理解成"航线"的代理变量。
树模型要刻画"某条航线的人更容易满意"需要很多次分裂；
直接给一列"这条航线的历史满意率"能让模型一步到位。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_keys(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """把若干个原始列拼成一个高基数类别列（用做分组键）。

    先把数值列变成字符串，保证拼接结果稳定。
    """
    parts = [df[k].astype(str) for k in keys]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def fit_encoding(keys: pd.Series, y: np.ndarray, prior: float, m: float) -> pd.Series:
    """在给定数据上拟合"类别 -> 平滑后目标均值"的映射表。"""
    stats = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (stats["sum"] + prior * m) / (stats["count"] + m)


def apply_encoding(keys: pd.Series, table: pd.Series, prior: float) -> np.ndarray:
    """把编码表应用到新数据；没见过的类别回退到全局均值 prior。"""
    return keys.map(table).fillna(prior).to_numpy(dtype=np.float64)
