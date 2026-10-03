"""Stuff+ model trainer.

Runs a single (target, model, feature_set, holdout_scheme) experiment with
leakage-free validation:

  pitcher   GroupKFold by pitcher_anon_id   -> generalizes to unseen pitchers
  season    train 2024+2025 / test 2026     -> chronological holdout
  park      train No+Medium / test Extreme   -> isolates the altitude effect

Arsenal features are always computed from the TRAINING fold only (cold-start
league-median prior for unseen pitchers).  Metrics are compared against a null
baseline (mean target by count) and written to results/<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, mean_squared_error, mean_absolute_error,
)
from sklearn.model_selection import GroupKFold

import features
import targets

# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def make_folds(df: pd.DataFrame, scheme: str, n_folds: int):
    """Return list of (train_idx, test_idx) over the FULL dataframe."""
    if scheme == "pitcher":
        groups = pd.factorize(df["pitcher_anon_id"])[0]
        gkf = GroupKFold(n_splits=n_folds)
        return list(gkf.split(np.arange(len(df)), groups=groups))
    if scheme == "season":
        train_idx = np.where(df["year"].isin([2024, 2025]).values)[0]
        test_idx = np.where((df["year"] == 2026).values)[0]
        return [(train_idx, test_idx)]
    if scheme == "park":
        train_idx = np.where(df["altitude_category"].isin(["No Altitude", "Medium Altitude"]).values)[0]
        test_idx = np.where((df["altitude_category"] == "Extreme Altitude").values)[0]
        return [(train_idx, test_idx)]
    raise ValueError(f"unknown holdout scheme: {scheme}")


# ---------------------------------------------------------------------------
# null baseline (mean target by count) -- the competition's reference model
# ---------------------------------------------------------------------------
def null_predict(train_df: pd.DataFrame, y_train, test_df: pd.DataFrame, regression: bool):
    """Predict the mean target within each count state (learned on train only)."""
    key_cols = ["Balls", "Strikes"]
    tr = train_df[key_cols].copy()
    tr["_y"] = y_train.values
    means = tr.groupby(["Balls", "Strikes"], observed=True)["_y"].mean()
    pred = test_df[key_cols].merge(
        means.rename("_p"), on=["Balls", "Strikes"], how="left"
    )["_p"]
    pred = pred.fillna(means.mean())
    return pred.values


# ---------------------------------------------------------------------------
# model factory
# ---------------------------------------------------------------------------
def make_model(name: str, regression: bool, n_jobs: int, xgb_overrides: dict | None = None):
    if name == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        kw = dict(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, tree_method="hist", n_jobs=n_jobs,
            random_state=0, verbosity=0,
        )
        kw.update(xgb_overrides or {})
        if regression:
            return XGBRegressor(**kw)
        return XGBClassifier(**kw, eval_metric="logloss")
    if name == "lgb":
        from lightgbm import LGBMClassifier, LGBMRegressor
        kw = dict(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_lambda=1.0, n_jobs=n_jobs, random_state=0, verbose=-1,
        )
        return LGBMRegressor(**kw) if regression else LGBMClassifier(**kw)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        if regression:
            return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                                early_stopping=True, random_state=0, alpha=1e-3)
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                             early_stopping=True, random_state=0, alpha=1e-3)
    if name in ("gam", "ensemble"):
        # handled inside run_experiment (need feature count / two models)
        return None
    raise ValueError(name)


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """Expected Calibration Error (ECE) for probabilistic predictions."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_pred >= lo) & (y_pred < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y_pred)) * abs(
            y_pred[m].mean() - y_true[m].mean()
        )
    return float(ece)


# ---------------------------------------------------------------------------
# core experiment
# ---------------------------------------------------------------------------
def run_experiment(target, model_name, feature_set, scheme, n_folds, n_jobs, tag,
                   save_oof=True, xgb_overrides=None):
    t0 = time.time()
    df = targets.build_targets(targets.load())
    regression = target in targets.REGRESSION_TARGETS
    ycol = targets.TARGET_COL[target]
    cat_levels = features.fit_categorical_levels(df)

    folds = make_folds(df, scheme, n_folds)
    fold_metrics = []
    oof_parts = []

    for fi, (train_idx, test_idx) in enumerate(folds):
        # target subset mask (swings / takes / batted / all)
        sub_mask = targets.TARGET_MASK[target]
        mask = np.ones(len(df), dtype=bool) if sub_mask is None else sub_mask(df).values
        tr = train_idx[mask[train_idx]]
        te = test_idx[mask[test_idx]]

        train_df, test_df = df.iloc[tr], df.iloc[te]
        y_train = train_df[ycol].astype(float).values
        y_test = test_df[ycol].astype(float).values

        # fastball baselines ALWAYS from the full training fold (all pitches)
        baseline_df = df.iloc[train_idx]
        X_train, feat_names = features.build_X(train_df, baseline_df, feature_set, cat_levels)
        X_test, _ = features.build_X(test_df, baseline_df, feature_set, cat_levels)

        model = make_model(model_name, regression, n_jobs, xgb_overrides)
        if model_name in ("mlp",):
            # MLP needs scaled features; fit scaler on TRAIN only (no leakage)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(sc.transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)
        if model_name == "ensemble":
            # xgb + lgb average -> lower variance, better generalization
            from xgboost import XGBClassifier, XGBRegressor
            from lightgbm import LGBMClassifier, LGBMRegressor
            if regression:
                m1 = make_model("xgb", True, n_jobs)
                m2 = make_model("lgb", True, n_jobs)
                m1.fit(X_train, y_train); m2.fit(X_train, y_train)
                pred = 0.5 * (m1.predict(X_test) + m2.predict(X_test))
            else:
                m1 = make_model("xgb", False, n_jobs)
                m2 = make_model("lgb", False, n_jobs)
                m1.fit(X_train, y_train); m2.fit(X_train, y_train)
                pred = 0.5 * (m1.predict_proba(X_test)[:, 1]
                              + m2.predict_proba(X_test)[:, 1])
        elif model_name == "gam":
            # interpretable spline model; subsample for tractability (pyGAM is
            # O(n·k^2)); fit one spline per (numeric) feature on a 30k sample.
            from pygam import LinearGAM, s
            Xs = X_train.sample(n=min(30000, len(X_train)), random_state=0)
            ys = pd.Series(y_train).loc[Xs.index].values
            terms = s(0)
            for j in range(1, Xs.shape[1]):
                terms += s(j)
            model = LinearGAM(terms, max_iter=50)
            model.fit(Xs.values, ys)
            pred = model.predict(X_test.values)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if not regression:
                pred = model.predict_proba(X_test)[:, 1]

        # null baseline
        null_pred = null_predict(train_df, train_df[ycol].astype(float), test_df, regression)

        m = {}
        if regression:
            m["rmse"] = float(np.sqrt(mean_squared_error(y_test, pred)))
            m["mae"] = float(mean_absolute_error(y_test, pred))
            m["bias"] = float(np.mean(pred) - np.mean(y_test))
            m["null_rmse"] = float(np.sqrt(mean_squared_error(y_test, null_pred)))
            m["rmse_ratio"] = m["rmse"] / max(m["null_rmse"], 1e-9)
        else:
            pclip = np.clip(pred, 1e-7, 1 - 1e-7)
            m["auc"] = float(roc_auc_score(y_test, pred))
            m["brier"] = float(brier_score_loss(y_test, pred))
            m["logloss"] = float(log_loss(y_test, pclip))
            m["ece"] = expected_calibration_error(y_test, pred)
            m["base_rate"] = float(np.mean(y_test))
            m["null_brier"] = float(brier_score_loss(y_test, null_pred))
            m["null_auc"] = float(roc_auc_score(y_test, null_pred))
        fold_metrics.append(m)

        if save_oof:
            oof_parts.append(pd.DataFrame({
                "PitchUID": test_df["PitchUID"].values,
                "y": y_test, "pred": pred,
                "pitcher": test_df["pitcher_anon_id"].values,
                "pitch_type": test_df["AutoPitchType"].values,
                "altitude": test_df["altitude_category"].values,
                "year": test_df["year"].values,
                "balls": test_df["Balls"].values,
                "strikes": test_df["Strikes"].values,
            }))

    # aggregate (mean + std across folds = reliability evidence)
    agg = {}
    std = {}
    for k in fold_metrics[0]:
        vals = [fm[k] for fm in fold_metrics]
        agg[k] = float(np.mean(vals))
        std[k] = float(np.std(vals))

    result = {
        "tag": tag,
        "target": target,
        "model": model_name,
        "features": feature_set,
        "holdout": scheme,
        "n_folds": len(folds),
        "n_features": len(feat_names),
        "fold_metrics": fold_metrics,
        "mean": agg,
        "std": std,
        "elapsed_s": round(time.time() - t0, 1),
    }
    fname = f"results/{tag}__{target}__{model_name}__{feature_set}__{scheme}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    if save_oof and oof_parts:
        oof = pd.concat(oof_parts, ignore_index=True)
        base = fname[:-5]  # strip .json
        oof.to_parquet(base + ".oof.parquet", index=False)
    print(json.dumps(result["mean"], indent=2))
    print(f"wrote {fname}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--features", default="pitching+")
    ap.add_argument("--holdout", default="pitcher")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--scale-pos-weight", type=float, default=None,
                    help="upweight the positive class (for rare-event targets like barrel)")
    args = ap.parse_args()
    overrides = {k: v for k, v in {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "scale_pos_weight": args.scale_pos_weight,
    }.items() if v is not None}
    run_experiment(args.target, args.model, args.features, args.holdout,
                   args.n_folds, args.n_jobs, args.tag, xgb_overrides=overrides or None)
