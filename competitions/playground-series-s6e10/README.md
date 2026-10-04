# Playground Series S6E10 — Predicting Airline Satisfaction

| 项目 | 内容 |
|---|---|
| 链接 | https://www.kaggle.com/competitions/playground-series-s6e10 |
| 竞赛 ID | 125224 |
| 任务 | 二分类：预测乘客是否满意（`satisfaction`） |
| 评测指标 | ROC AUC（越大越好） |
| 数据截止 | 2026-10-31 23:59 UTC |
| 每日提交上限 | 10 次 |
| 最终需选定 | 2 个提交用于评分 |
| 每行 id 列 | `id` |
| 排行榜公开比例 | 20% |
| 当前规模 | 592 队 / 606 人 / 3440 次提交 |

## 结果

| 阶段 | 本地 OOF AUC | 线上公开榜 |
|---|---|---|
| LightGBM 基线（原始特征） | 0.958910 | 0.95843 |
| + 航线距离目标编码 + 三模型融合 | 0.960505 | 0.96021 |
| **+ 十七模型贪心融合（最佳）** | **0.960738** | **0.96029** |

公开榜名次从 346 / 595 提升到 **248 / 612**。
榜单中位数 0.95898、75% 分位 0.96109、第 1 名 0.96167。

### 一句话结论

十个尝试方向里只有一个真正有效：**用精确的 `Flight Distance` 做折内目标编码**
（+0.0014）。它成立的原因是信号藏在 3474 个距离取值这个高基数结构里，
而树模型很难自己构造出这种表示。其余方向（外部数据、手工特征工程、
更大容量、超参搜索）全部在噪声范围内，详见 `docs/`。

## 为什么是这个问题

Kaggle Playground 系列每题都由一个真实公开数据集"合成"而来：用生成模型
从原始数据学到分布后，再采样出更大的训练集。本题的原始数据集是经典的
*Airline Passenger Satisfaction*（航空乘客满意度调查）。

合成数据带来两个典型特征，直接决定了做法：

1. **信号被"平滑"了**：合成器抹掉了一部分原始噪声结构，所以这类题目
   上限通常很高（本题公开榜头部已经在 0.9616+），且**简单模型就很强**。
2. **列基本都是有序类别（1–5 分打分）**，所以树模型几乎必然优于线性模型；
   而"要不要把打分列当连续变量"是一个值得实验的点。

## 我的方案（迭代路线）

按"先跑通、再变强"的顺序推进，每一步都记录在 `docs/` 里：

- **M1 打通链路**：装环境 → 下数据 → 看数据 → 交一个能得分的基线。
- **M2 强基线**：LightGBM / XGBoost / CatBoost 各跑 5 折交叉验证，
  用 CV 分数（而不是拍脑袋）挑模型。
- **M3 特征工程**：针对"打分列"和"延误时间"做针对性处理，逐个用 CV 验证。
- **M4 融合**：多模型 + 多随机种子的加权/堆叠融合。
- **M5 收尾**：确认提交格式、按 10 次/天额度谨慎提交、选定最终提交。

## 目录

- `docs/` — 逐步思路与实验结论
- `src/` — 脚本，按编号顺序可复现
- `configs/` — 实验参数
- `data/`、`outputs/` — 不提交（见根目录 `.gitignore`）

## 复现步骤

```bash
cd competitions/playground-series-s6e10

# 0. 环境（需要 ~/.kaggle/access_token 或 kaggle.json）
pip install -r requirements.txt

# 1. 数据
python src/00_download_data.py        # 比赛数据 -> data/raw/
python src/03_download_external.py    # 外部原始数据 -> data/external/
python src/01_eda.py                  # 数据探查

# 2. 基线（两条腿：CV 分数 + 一次线上校准）
python src/02_baseline_lgbm.py        # OOF 0.958910

# 3. 表示：航线距离目标编码（本项目唯一的有效杠杆）
python src/09_key_scan.py             # 穷举 231 个分组键，圈定候选
python src/05_target_encoding_exp.py dist                 # OOF 0.960273
python src/05_target_encoding_exp.py dist --smooth 2      # OOF 0.960325

# 4. 超参数（结论：收益在噪声量级）
python src/08_optuna.py --trials 40 --n-splits 3 --timeout 2400

# 5. 多模型
python src/06_models.py lgb --lgb-params outputs/optuna_lgb_te_best.json --tag _tuned
python src/06_models.py xgb cat lgb_et lgb_goss xgb_deep

# 6. 贪心融合 + 生成提交文件
python src/07_blend.py lgb_tuned te_dist_m2 te_all te_dist_ob_cls_cat \
    te_dist_m5 te_dist xgb te_dist_m50 te_dist_inner5 te_dist_ob_cls \
    te_dist_ob te_all_nofd te_fine cat lgb_et lgb_goss xgb_deep

# 7. 提交
kaggle competitions submit -c playground-series-s6e10 \
    -f outputs/submission_blend.csv -m "message"
```

注意：`04_experiments.py`（外部数据/特征工程/超参对比）是探索性脚本，
结论已固化到文档，不需要重跑。
