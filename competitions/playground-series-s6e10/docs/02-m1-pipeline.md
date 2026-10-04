# 02 · M1 打通链路

## 完成的动作与结论

### 1. 凭据

Kaggle 已经在用新版令牌机制：`~/.kaggle/access_token`（前缀 `KGAT_`），
不再需要老的 `kaggle.json`（username + key）。CLI v2 会自动读取。
认证通过，账号 `pengfeili0811`。

> 记录这个是因为踩过一次坑：老版本 kaggle CLI 只认 `kaggle.json`，
> 如果不升级会一直报"找不到凭据"。

### 2. 比赛数据文件清单

```
name                        size
sample_submission.csv    8.1 MB
test.csv                27.2 MB
train.csv               67.2 MB
```

从文件体积反推样本量：train 约是 test 的 2.5 倍。参考同类 Playground
竞赛的惯例，预计 train 在 **100 万行级别**、test 在 **40 万行级别**。
这个规模意味着：CPU 上跑树模型完全够用，不需要 GPU；
但**每次实验要认真估时**，不能随意跑几十组参数。

### 3. 遇到的阻塞：403 Forbidden

下载时报：

```
403 Client Error: Forbidden for url:
https://api.kaggle.com/v1/competitions.CompetitionApiService/DownloadDataFiles
```

**判断依据**：同一个账号、同一个令牌，`competitions files`（列文件）能成功，
只有 `download` 被拒。这不是认证问题，而是**权限问题**——
Kaggle 的规则是：必须先在这场比赛页面点 **Join Competition** 并接受比赛规则
（`rulesRequired: true`），才能下载数据和提交结果。

这一步**无法用 API 完成**，必须人工在浏览器里点一下。这是 Kaggle 有意为之：
让它成为一个有法律意义的"同意"动作。

### 4. 为什么这个设计对参赛者重要

- 接受规则意味着你同意：不能使用外部数据（除非规则允许）、
  不能组队超过 3 人、提交必须真实。
- 本题数据是合成的，且规则通常限制"不得使用原始数据集"。
  这一点后续做特征工程时要留意。

## 下一步

1. 人工加入比赛（阻塞中）
2. 下载数据 → 运行 `01_eda.py`，把数据形状、列类型、缺失、正类比例记录下来
3. 跑 `02_baseline_lgbm.py`，拿到第一个 OOF AUC
4. 提交一次，核对本地 CV 与线上分数的偏差——这一步决定后面能不能信任 CV
