"""特征工程。

设计原则：**每个特征都要能说出一个理由**，不做"反正加了试试"的堆砌。
每加一个特征都要用 CV 验证增益，否则回退。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 13 个 0-5 分的满意度打分项（乘客对各个环节的评分）
RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """在原始列之外补充派生特征。返回新 DataFrame，不修改入参。"""
    df = df.copy()

    ratings = df[RATING_COLS]

    # --- 打分项的整体聚合 ---
    # 理由：单个评分噪声大，但"总体满意度水平"这个信号很稳。
    # 一个处处打 4 分的人和一个处处打 2 分的人，满意度必然不同。
    df["rating_sum"] = ratings.sum(axis=1)
    df["rating_mean"] = ratings.mean(axis=1)
    df["rating_std"] = ratings.std(axis=1)
    df["rating_min"] = ratings.min(axis=1)
    df["rating_max"] = ratings.max(axis=1)

    # 打了 0 分的项目数。0 在原始问卷里表示"未使用/不适用"，
    # 而不是"最差"。所以 0 的数量本身可能携带"这位乘客用了哪些服务"的信息。
    df["rating_zero_count"] = (ratings == 0).sum(axis=1)

    # 高分项数量：直接数"给了 4 分或 5 分"的环节有几个。
    # 这比均值更贴近"AUC 只关心排序"的需求。
    df["rating_high_count"] = (ratings >= 4).sum(axis=1)
    df["rating_low_count"] = (ratings <= 2).sum(axis=1)

    # --- 延误相关 ---
    dep = df["Departure Delay in Minutes"]
    arr = df["Arrival Delay in Minutes"]

    df["delay_total"] = dep + arr.fillna(0)
    df["delay_diff"] = arr.fillna(0) - dep
    # 长尾分布取对数，压缩极端值的影响
    df["delay_total_log"] = np.log1p(df["delay_total"])
    df["is_delayed"] = (dep > 0).astype(np.int8)

    # 到达延误缺失本身也是一个信号（通常对应短途航班或数据缺失模式）
    df["arr_delay_missing"] = arr.isna().astype(np.int8)

    # --- 交互 / 比例 ---
    # 每单位飞行距离的评分：长途航班更容易不满意，用比例把这两个因素解耦
    df["rating_per_distance"] = df["rating_sum"] / (df["Flight Distance"] + 1)
    # 年龄与飞行距离的交互：商务常旅客 vs 首次出行的家庭游客
    df["age_x_distance"] = df["Age"] * df["Flight Distance"] / 1000.0

    return df


def add_features_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """只加最核心的一组特征，用于做增量对比（消融实验）。"""
    df = df.copy()
    ratings = df[RATING_COLS]
    df["rating_mean"] = ratings.mean(axis=1)
    df["rating_std"] = ratings.std(axis=1)
    df["rating_high_count"] = (ratings >= 4).sum(axis=1)
    df["delay_total_log"] = np.log1p(df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"].fillna(0))
    df["arr_delay_missing"] = df["Arrival Delay in Minutes"].isna().astype(np.int8)
    return df
