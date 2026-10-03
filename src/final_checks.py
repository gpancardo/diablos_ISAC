"""Fast final checks: (1) n_estimators over/under-fit sweep, (2) season holdout whiff.

The full early-stopping sweep was killed (n_estimators=3000 ran forever); this
answers the same question cheaply: does tree count overfit, and does whiff
generalize across seasons?
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import features
import targets


def load_whiff():
    df = targets.build_targets(targets.load())
    sub = df[targets.TARGET_MASK["whiff"](df)].reset_index(drop=True)
    return sub


def n_estimators_sweep(n_folds=5, seed=0):
    """whiff xgb stuff, GroupKFold pitcher: AUC at n_estimators in {150,300,600,1200}."""
    sub = load_whiff()
    cat_levels = features.fit_categorical_levels(sub)
    y = sub["y_whiff"].astype(float).values
    groups = pd.factorize(sub["pitcher_anon_id"])[0]
    gkf = GroupKFold(n_splits=n_folds)
    NEs = [150, 300, 600, 1200]
    aucs = {n: [] for n in NEs}
    for tr, te in gkf.split(np.arange(len(sub)), groups=groups):
        base = sub.iloc[tr]
        X_tr, _ = features.build_X(sub.iloc[tr], base, "stuff", cat_levels)
        X_te, _ = features.build_X(sub.iloc[te], base, "stuff", cat_levels)
        y_tr, y_te = y[tr], y[te]
        for n in NEs:
            m = XGBClassifier(n_estimators=n, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              tree_method="hist", n_jobs=4, random_state=seed,
                              verbosity=0, eval_metric="logloss")
            m.fit(X_tr, y_tr)
            aucs[n].append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    return {str(n): {"auc": float(np.mean(v)), "std": float(np.std(v))} for n, v in aucs.items()}


def season_holdout():
    sub = load_whiff()
    cat_levels = features.fit_categorical_levels(sub)
    y = sub["y_whiff"].astype(float).values
    tr = np.where(sub["year"].isin([2024, 2025]).values)[0]
    te = np.where((sub["year"] == 2026).values)[0]
    base = sub.iloc[tr]
    X_tr, _ = features.build_X(sub.iloc[tr], base, "stuff", cat_levels)
    X_te, _ = features.build_X(sub.iloc[te], base, "stuff", cat_levels)
    y_tr, y_te = y[tr], y[te]

    m1 = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                       tree_method="hist", n_jobs=4, random_state=0, verbosity=0,
                       eval_metric="logloss")
    m1.fit(X_tr, y_tr)
    p1 = m1.predict_proba(X_te)[:, 1]

    m2 = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                        n_jobs=4, random_state=0, verbose=-1)
    m2.fit(X_tr, y_tr)
    p2 = m2.predict_proba(X_te)[:, 1]
    p_ens = 0.5 * (p1 + p2)

    return {
        "xgb_auc": float(roc_auc_score(y_te, p1)),
        "lgb_auc": float(roc_auc_score(y_te, p2)),
        "ensemble_auc": float(roc_auc_score(y_te, p_ens)),
        "ensemble_brier": float(brier_score_loss(y_te, p_ens)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_base_rate": float(np.mean(y_te)),
    }


if __name__ == "__main__":
    print("=== n_estimators sweep (whiff, pitcher holdout) ===", flush=True)
    nes = n_estimators_sweep()
    print(json.dumps(nes, indent=2), flush=True)

    print("=== season holdout whiff (train 24+25 / test 2026) ===", flush=True)
    sh = season_holdout()
    print(json.dumps(sh, indent=2), flush=True)

    with open("results/final_checks.json", "w") as f:
        json.dump({"n_estimators": nes, "season": sh}, f, indent=2)
    print("wrote results/final_checks.json", flush=True)
