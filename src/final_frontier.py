"""Final-frontier experiments: the levers NOT yet pulled for precision/generalization.

Answers 5 concrete questions (all whiff-first, the flagship Stuff+ target):

  Q1. Early stopping / n_estimators   -- is the fixed 600 trees over/under-fitting?
  Q2. Regularization sweep            -- does heavier reg improve generalization?
  Q3. Season holdout                  -- the honest cross-year ceiling (train 24+25 / test 26)
  Q4. Bootstrap CI (clustered)        -- is 0.7449 vs 0.75 actually different?
  Q5. Weighted stack (learned blend)  -- beats the 0.5/0.5 average?

All leak-free: early-stopping validation split by pitcher (disjoint from train
and from the outer holdout); stacking meta-learner cross-validated by pitcher.
"""
from __future__ import annotations

import json
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

import features
import targets

T0 = time.time()


def load_whiff_df():
    df = targets.build_targets(targets.load())
    sub = df[targets.TARGET_MASK["whiff"](df)].reset_index(drop=True)
    return sub


def build_stuff_X(df, baseline_df, cat_levels):
    X, names = features.build_X(df, baseline_df, "stuff", cat_levels)
    return X, names


# ---------------------------------------------------------------------------
# Q1 + Q2 : early stopping and regularization on the pitcher holdout
# ---------------------------------------------------------------------------
def run_early_stopping(n_folds=10, seed=0):
    """GroupKFold by pitcher; within each fold, hold out 20% of TRAIN pitchers for
    early stopping.  Compare fixed-600 vs early-stopped vs heavy-regularized."""
    from xgboost import XGBClassifier

    df = load_whiff_df()
    cat_levels = features.fit_categorical_levels(df)
    y_all = df["y_whiff"].astype(float).values
    groups = pd.factorize(df["pitcher_anon_id"])[0]
    gkf = GroupKFold(n_splits=n_folds)

    variants = {
        "fixed600": dict(n_estimators=600, early_stop=False, reg="base"),
        "early_stop": dict(n_estimators=3000, early_stop=True, reg="base"),
        "early_stop+reg": dict(n_estimators=3000, early_stop=True, reg="strong"),
    }
    agg = {k: {"auc": [], "brier": [], "best_iter": []} for k in variants}

    for tr_idx, te_idx in gkf.split(np.arange(len(df)), groups=groups):
        train_idx = np.asarray(tr_idx)
        # split TRAIN pitchers into inner train / early-stop valid (disjoint pitchers)
        tr_groups = groups[train_idx]
        uniq, uniq_inv = np.unique(tr_groups, return_inverse=True)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(uniq))
        n_valid_g = max(1, int(0.2 * len(uniq)))
        valid_g = set(perm[:n_valid_g])
        in_train = np.array([uniq_inv[i] not in valid_g for i in range(len(train_idx))])
        itr = train_idx[in_train]
        iva = train_idx[~in_train]

        base_df = df.iloc[itr]
        X_tr, names = build_stuff_X(df.iloc[itr], base_df, cat_levels)
        X_va, _ = build_stuff_X(df.iloc[iva], base_df, cat_levels)
        X_te, _ = build_stuff_X(df.iloc[te_idx], base_df, cat_levels)
        y_tr = y_all[itr]
        y_va = y_all[iva]
        y_te = y_all[te_idx]

        for vname, cfg in variants.items():
            kw = dict(n_estimators=cfg["n_estimators"], learning_rate=0.05,
                      max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, reg_lambda=1.0, tree_method="hist",
                      n_jobs=2, random_state=seed, verbosity=0,
                      eval_metric="logloss")
            if cfg["reg"] == "strong":
                kw.update(learning_rate=0.03, max_depth=4, subsample=0.7,
                          colsample_bytree=0.5, min_child_weight=10,
                          reg_lambda=10.0, gamma=0.5)
            if cfg["early_stop"]:
                # xgboost >=2.0: early_stopping_rounds is a constructor arg
                kw["early_stopping_rounds"] = 50
            m = XGBClassifier(**kw)
            if cfg["early_stop"]:
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                agg[vname]["best_iter"].append(int(m.best_iteration))
            else:
                m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)[:, 1]
            agg[vname]["auc"].append(roc_auc_score(y_te, p))
            agg[vname]["brier"].append(brier_score_loss(y_te, p))

    out = {}
    for vname, d in agg.items():
        out[vname] = {
            "auc_mean": float(np.mean(d["auc"])),
            "auc_std": float(np.std(d["auc"])),
            "brier_mean": float(np.mean(d["brier"])),
            "best_iter_median": float(np.median(d["best_iter"])) if d["best_iter"] else None,
        }
    return out


# ---------------------------------------------------------------------------
# Q3 : season holdout for whiff (train 2024+2025, test 2026)
# ---------------------------------------------------------------------------
def run_season_whiff():
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    df = load_whiff_df()
    cat_levels = features.fit_categorical_levels(df)
    y_all = df["y_whiff"].astype(float).values

    train_idx = np.where(df["year"].isin([2024, 2025]).values)[0]
    test_idx = np.where((df["year"] == 2026).values)[0]
    base_df = df.iloc[train_idx]
    X_tr, _ = build_stuff_X(df.iloc[train_idx], base_df, cat_levels)
    X_te, _ = build_stuff_X(df.iloc[test_idx], base_df, cat_levels)
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    res = {}
    m1 = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                       tree_method="hist", n_jobs=2, random_state=0, verbosity=0,
                       eval_metric="logloss")
    m1.fit(X_tr, y_tr)
    res["xgb_auc"] = float(roc_auc_score(y_te, m1.predict_proba(X_te)[:, 1]))
    res["xgb_brier"] = float(brier_score_loss(y_te, m1.predict_proba(X_te)[:, 1]))

    m2 = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                        n_jobs=2, random_state=0, verbose=-1)
    m2.fit(X_tr, y_tr)
    p = 0.5 * (m1.predict_proba(X_te)[:, 1] + m2.predict_proba(X_te)[:, 1])
    res["ensemble_auc"] = float(roc_auc_score(y_te, p))
    res["ensemble_brier"] = float(brier_score_loss(y_te, p))
    res["n_test"] = int(len(y_te))
    res["base_rate"] = float(np.mean(y_te))
    return res


# ---------------------------------------------------------------------------
# Q4 : clustered bootstrap CI on the existing whiff OOF
# ---------------------------------------------------------------------------
def bootstrap_ci(oof_path, n_boot=1000, seed=0):
    oof = pd.read_parquet(oof_path)
    rng = np.random.RandomState(seed)
    pitchers = oof["pitcher"].unique()
    # precompute pitcher -> row indices (numpy), avoid per-pitcher pandas filters
    pvals = oof["pitcher"].values
    p_index = {p: np.where(pvals == p)[0] for p in pitchers}
    y = oof["y"].values
    pred = oof["pred"].values
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        sample_pitchers = rng.choice(pitchers, size=len(pitchers), replace=True)
        idx = np.concatenate([p_index[p] for p in sample_pitchers])
        aucs[b] = roc_auc_score(y[idx], pred[idx])
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"auc_mean": float(aucs.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "point": float(roc_auc_score(y, pred))}


# ---------------------------------------------------------------------------
# Q5 : learned stack (xgb + lgb) -- meta-learner CV'd by pitcher
# ---------------------------------------------------------------------------
def run_stack():
    xgb_oof = pd.read_parquet("results/r2__whiff__xgb__stuff__pitcher.oof.parquet")
    lgb_oof = pd.read_parquet("results/r2__whiff__lgb__stuff__pitcher.oof.parquet")
    m = xgb_oof[["PitchUID", "y", "pitcher", "pred"]].merge(
        lgb_oof[["PitchUID", "pred"]].rename(columns={"pred": "pred_lgb"}), on="PitchUID")
    m = m.rename(columns={"pred": "pred_xgb"})
    y = m["y"].values
    X = m[["pred_xgb", "pred_lgb"]].values
    groups = pd.factorize(m["pitcher"])[0]

    # baseline: 0.5/0.5 average
    avg = 0.5 * (X[:, 0] + X[:, 1])
    base_auc = roc_auc_score(y, avg)

    # learned stack, CV'd by pitcher (no leakage)
    gkf = GroupKFold(n_splits=10)
    stack_pred = np.zeros(len(y))
    for tr, te in gkf.split(np.arange(len(y)), groups=groups):
        lr = LogisticRegression(C=10.0, max_iter=1000)
        lr.fit(X[tr], y[tr])
        stack_pred[te] = lr.predict_proba(X[te])[:, 1]
    stack_auc = roc_auc_score(y, stack_pred)

    # learned weights on full OOF (report, but CV number is the honest one)
    lr = LogisticRegression(C=10.0, max_iter=1000).fit(X, y)
    w_xgb, w_lgb = lr.coef_[0]
    return {"avg_auc": float(base_auc), "stack_auc": float(stack_auc),
            "w_xgb": float(w_xgb), "w_lgb": float(w_lgb)}


if __name__ == "__main__":
    print("=== Q4 bootstrap CI (existing whiff xgb stuff OOF) ===", flush=True)
    ci = bootstrap_ci("results/r2__whiff__xgb__stuff__pitcher.oof.parquet")
    print(json.dumps(ci, indent=2), flush=True)

    print("=== Q5 learned stack (xgb+lgb) ===", flush=True)
    st = run_stack()
    print(json.dumps(st, indent=2), flush=True)

    print("=== Q1+Q2 early stopping + regularization (whiff, pitcher 10-fold) ===",
          flush=True)
    es = run_early_stopping()
    print(json.dumps(es, indent=2), flush=True)

    print("=== Q3 season holdout whiff (train 24+25 / test 26) ===", flush=True)
    se = run_season_whiff()
    print(json.dumps(se, indent=2), flush=True)

    summary = {"ci": ci, "stack": st, "early_stop": es, "season": se,
               "elapsed_s": round(time.time() - T0, 1)}
    with open("results/final_frontier.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/final_frontier.json", flush=True)
