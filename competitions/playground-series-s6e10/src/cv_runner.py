"""可复用的交叉验证运行器。

为什么单独抽一个模块：
  进入 M3 之后要做几十组对比实验（不同特征、不同模型、不同权重）。
  如果每次实验都复制一遍训练循环，很快就会"两组实验用了不同的折/不同的早停规则"，
  结论也就不再可比较。这里把规则**固定下来**：

    - 折的划分永远来自 common.make_folds（同一个随机种子）；
    - OOF 永远只在我们自己的比赛训练集上计算；
    - 早停永远只看我们自己的验证折（即使训练时混入了外部数据）。

  最后一条尤其重要：外部数据的分布和比赛数据不完全一样，
  如果拿外部数据来做早停，选出的迭代次数是"对原始数据最好"而非"对比赛最好"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import SEED, make_folds


def run_lgb_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    cat_cols: list[str],
    X_ext: pd.DataFrame | None = None,
    y_ext: np.ndarray | None = None,
    ext_weight: float = 1.0,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    verbose: bool = True,
    n_splits: int = 5,
) -> dict:
    """跑一次 5 折 LightGBM，返回 OOF / 测试集预测 / 分数。"""
    import lightgbm as lgb

    folds = make_folds(y, n_splits=n_splits)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        w_tr = None

        if X_ext is not None and ext_weight > 0:
            # 外部数据全部加入每一折的训练集（它不是验证集的一部分）
            X_tr = pd.concat([X_tr, X_ext], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, y_ext])
            # 降权：让外部数据对损失函数的贡献小于比赛数据
            w_tr = np.concatenate([
                np.ones(len(tr_idx), dtype=np.float64),
                np.full(len(X_ext), float(ext_weight), dtype=np.float64),
            ])

        dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            params, dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)
        fold_scores.append(roc_auc_score(y[va_idx], oof[va_idx]))
        best_iters.append(model.best_iteration)
        if verbose:
            print(f"    fold {i}: AUC={fold_scores[-1]:.6f}  best_iter={model.best_iteration}")

    auc = roc_auc_score(y, oof)
    return {
        "oof": oof,
        "test_pred": test_pred,
        "auc": auc,
        "fold_scores": fold_scores,
        "best_iters": best_iters,
    }


def run_xgb_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    X_ext: pd.DataFrame | None = None,
    y_ext: np.ndarray | None = None,
    ext_weight: float = 1.0,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    verbose: bool = True,
) -> dict:
    """同样规则的 XGBoost 版本，保证与 LightGBM 的折完全一致、可直接融合。"""
    import xgboost as xgb

    folds = make_folds(y)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        w_tr = None
        if X_ext is not None and ext_weight > 0:
            X_tr = pd.concat([X_tr, X_ext], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, y_ext])
            w_tr = np.concatenate([
                np.ones(len(tr_idx)), np.full(len(X_ext), float(ext_weight)),
            ])

        dtr = xgb.DMatrix(X_tr, y_tr, weight=w_tr, enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], y[va_idx], enable_categorical=True)

        model = xgb.train(
            params, dtr,
            num_boost_round=num_boost_round,
            evals=[(dva, "valid")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )

        oof[va_idx] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / len(folds)
        fold_scores.append(roc_auc_score(y[va_idx], oof[va_idx]))
        best_iters.append(model.best_iteration + 1)
        if verbose:
            print(f"    fold {i}: AUC={fold_scores[-1]:.6f}  best_iter={best_iters[-1]}")

    auc = roc_auc_score(y, oof)
    return {
        "oof": oof, "test_pred": test_pred, "auc": auc,
        "fold_scores": fold_scores, "best_iters": best_iters,
    }


def run_lgb_cv_te(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    cat_cols: list[str],
    te_specs: list[tuple[str, np.ndarray, np.ndarray]],
    smooth: float = 20.0,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    verbose: bool = True,
    n_splits: int = 5,
    inner_folds: int = 1,
) -> dict:
    """带折内目标编码的 LightGBM 交叉验证。

    te_specs 里每一项是 (特征名, 训练集分组键, 测试集分组键)，
    分组键是与 X / X_test **按位置对齐**的字符串数组。
    分组键本身不依赖目标，所以可以在折外一次算好；
    编码表则必须在每折内部单独拟合，否则会泄漏。

    把分组键和特征矩阵分开传，是为了支持"用某列造编码、但不把该列本身喂给模型"
    这类消融实验（例如用 Flight Distance 造航线编码，同时移除原始距离列）。
    """
    import lightgbm as lgb
    from target_encoding import apply_encoding, fit_encoding

    train_keys = {name: pd.Series(tr_keys) for name, tr_keys, _ in te_specs}
    test_keys = {name: pd.Series(te_keys) for name, _, te_keys in te_specs}
    te_cols = [name for name, _, _ in te_specs]

    folds = make_folds(y, n_splits=n_splits)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_te = X_test.copy()
        y_tr, y_va = y[tr_idx], y[va_idx]
        prior = float(y_tr.mean())

        for name in te_cols:
            k_tr = train_keys[name]
            table = fit_encoding(k_tr.iloc[tr_idx], y_tr, prior, smooth)
            X_va[name] = apply_encoding(k_tr.iloc[va_idx], table, prior)
            X_te[name] = apply_encoding(test_keys[name], table, prior)

            if inner_folds > 1:
                # 折内再折：训练行的编码值由"不含自己"的那部分数据算出来，
                # 这样这个特征在训练集和验证集上的"好用程度"一致，
                # 模型不会因为训练时它显得更准而过度依赖它。
                enc_tr = np.zeros(len(tr_idx))
                for itr, iva in make_folds(y_tr, n_splits=inner_folds, seed=SEED + 1):
                    inner_table = fit_encoding(
                        k_tr.iloc[tr_idx[itr]], y_tr[itr], float(y_tr[itr].mean()), smooth,
                    )
                    enc_tr[iva] = apply_encoding(
                        k_tr.iloc[tr_idx[iva]], inner_table, float(y_tr[itr].mean()),
                    )
                X_tr[name] = enc_tr
            else:
                X_tr[name] = apply_encoding(k_tr.iloc[tr_idx], table, prior)

        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
        )

        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_pred += model.predict(X_te, num_iteration=model.best_iteration) / len(folds)
        fold_scores.append(roc_auc_score(y_va, oof[va_idx]))
        best_iters.append(model.best_iteration)
        if verbose:
            print(f"    fold {i}: AUC={fold_scores[-1]:.6f}  best_iter={model.best_iteration}")

    return {
        "oof": oof, "test_pred": test_pred, "auc": roc_auc_score(y, oof),
        "fold_scores": fold_scores, "best_iters": best_iters,
    }


def run_cv_te(
    model: str,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    cat_cols: list[str],
    te_specs: list[tuple[str, np.ndarray, np.ndarray]],
    smooth: float = 5.0,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    verbose: bool = True,
    n_splits: int = 5,
    inner_folds: int = 1,
) -> dict:
    """统一接口：在"目标编码 + 指定模型"下跑一次交叉验证。

    支持 model in {"lgb", "xgb", "cat"}。三个模型共用同一套折划分与同一套
    目标编码流程，因此产出的 OOF 可以直接拿去融合。

    不同库对类别特征的处理方式不一样，这是这里最容易出错的地方：
      - LightGBM：内部能识别 pandas 的 category 类型；
      - XGBoost：需要 enable_categorical=True；
      - CatBoost：要求类别列是字符串/整数，pandas category 它会报错，
        所以这里显式转成 str。三者的**编码含义**必须一致，
        因此类别字典在 split_features 里就已经统一好了。
    """
    from target_encoding import apply_encoding, fit_encoding

    train_keys = {name: pd.Series(tr_keys) for name, tr_keys, _ in te_specs}
    test_keys = {name: pd.Series(te_keys) for name, _, te_keys in te_specs}
    te_cols = [name for name, _, _ in te_specs]

    def make_frame(src: pd.DataFrame) -> pd.DataFrame:
        """按模型的需要调整类别列的 dtype。

        - LightGBM / XGBoost：保留 pandas 的 category 类型（XGBoost 需要
          enable_categorical=True 才认这种列）；
        - CatBoost：不接受 category 类型，必须给字符串或整数。
        """
        out = src.copy()
        if model == "cat":
            for c in cat_cols:
                out[c] = out[c].astype(str)
        return out

    X = make_frame(X)
    X_test = make_frame(X_test)

    folds = make_folds(y, n_splits=n_splits)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_te = X_test.copy()
        y_tr, y_va = y[tr_idx], y[va_idx]
        prior = float(y_tr.mean())

        for name in te_cols:
            k_tr = train_keys[name]
            table = fit_encoding(k_tr.iloc[tr_idx], y_tr, prior, smooth)
            X_va[name] = apply_encoding(k_tr.iloc[va_idx], table, prior)
            X_te[name] = apply_encoding(test_keys[name], table, prior)

            if inner_folds > 1:
                # 折内再折：训练行的编码值由"不含自己"的那部分数据算出来，
                # 这样这个特征在训练集和验证集上的"好用程度"一致，
                # 模型不会因为训练时它显得更准而过度依赖它。
                enc_tr = np.zeros(len(tr_idx))
                for itr, iva in make_folds(y_tr, n_splits=inner_folds, seed=SEED + 1):
                    inner_table = fit_encoding(
                        k_tr.iloc[tr_idx[itr]], y_tr[itr], float(y_tr[itr].mean()), smooth,
                    )
                    enc_tr[iva] = apply_encoding(
                        k_tr.iloc[tr_idx[iva]], inner_table, float(y_tr[itr].mean()),
                    )
                X_tr[name] = enc_tr
            else:
                X_tr[name] = apply_encoding(k_tr.iloc[tr_idx], table, prior)

        if model == "lgb":
            import lightgbm as lgb
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
            dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
            est = lgb.train(
                params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
            )
            va_pred = est.predict(X_va, num_iteration=est.best_iteration)
            te_pred = est.predict(X_te, num_iteration=est.best_iteration)
            best_iter = est.best_iteration

        elif model == "xgb":
            import xgboost as xgb
            dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
            dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
            dte = xgb.DMatrix(X_te, enable_categorical=True)
            est = xgb.train(
                params, dtr, num_boost_round=num_boost_round, evals=[(dva, "valid")],
                early_stopping_rounds=early_stopping, verbose_eval=False,
            )
            rng = (0, est.best_iteration + 1)
            va_pred = est.predict(dva, iteration_range=rng)
            te_pred = est.predict(dte, iteration_range=rng)
            best_iter = est.best_iteration + 1

        elif model == "cat":
            from catboost import CatBoostClassifier, Pool
            ptr = Pool(X_tr, y_tr, cat_features=cat_cols)
            pva = Pool(X_va, y_va, cat_features=cat_cols)
            pte = Pool(X_te, cat_features=cat_cols)
            est = CatBoostClassifier(**params)
            est.fit(ptr, eval_set=pva, use_best_model=True, verbose=False)
            va_pred = est.predict_proba(pva)[:, 1]
            te_pred = est.predict_proba(pte)[:, 1]
            best_iter = int(est.get_best_iteration() or 0)

        else:
            raise ValueError(f"未知模型：{model}")

        oof[va_idx] = va_pred
        test_pred += te_pred / len(folds)
        fold_scores.append(roc_auc_score(y_va, va_pred))
        best_iters.append(int(best_iter))
        if verbose:
            print(f"    fold {i}: AUC={fold_scores[-1]:.6f}  best_iter={best_iters[-1]}")

    return {
        "oof": oof, "test_pred": test_pred, "auc": roc_auc_score(y, oof),
        "fold_scores": fold_scores, "best_iters": best_iters,
    }
