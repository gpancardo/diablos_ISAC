"""Fold-aware hyperparameter search (pitcher GroupKFold).

Searches a small xgboost grid and reports the best params + CV metric vs the
default config.  Fold-aware: params are chosen on training folds only, so no
test information leaks into the search.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

import features
import targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", default="stuff")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--n-trees", type=int, default=400)
    args = ap.parse_args()

    from xgboost import XGBClassifier, XGBRegressor

    df = targets.build_targets(targets.load())
    regression = args.target in targets.REGRESSION_TARGETS
    ycol = targets.TARGET_COL[args.target]
    cat_levels = features.fit_categorical_levels(df)

    sub_mask = targets.TARGET_MASK[args.target]
    mask = np.ones(len(df), bool) if sub_mask is None else sub_mask(df).values
    df = df[mask].reset_index(drop=True)
    y = df[ycol].astype(float).values
    groups = pd.factorize(df["pitcher_anon_id"])[0]

    gkf = GroupKFold(n_splits=args.n_folds)
    folds = list(gkf.split(np.arange(len(df)), groups=groups))

    grid = [
        dict(max_depth=4, learning_rate=0.08, min_child_weight=5),
        dict(max_depth=6, learning_rate=0.05, min_child_weight=5),  # default
        dict(max_depth=8, learning_rate=0.03, min_child_weight=5),
        dict(max_depth=6, learning_rate=0.03, min_child_weight=1),
        dict(max_depth=8, learning_rate=0.08, min_child_weight=1),
        dict(max_depth=5, learning_rate=0.05, min_child_weight=20),
    ]

    results = []
    for params in grid:
        scores = []
        for tr, te in folds:
            Xtr, _ = features.build_X(df.iloc[tr], df.iloc[tr], args.features, cat_levels)
            Xte, _ = features.build_X(df.iloc[te], df.iloc[tr], args.features, cat_levels)
            kw = dict(n_estimators=args.n_trees, subsample=0.8, colsample_bytree=0.8,
                      tree_method="hist", n_jobs=2, random_state=0, verbosity=0, **params)
            if regression:
                m = XGBRegressor(**kw)
                m.fit(Xtr, y[tr])
                p = m.predict(Xte)
                scores.append(float(np.sqrt(mean_squared_error(y[te], p))))
            else:
                m = XGBClassifier(**kw, eval_metric="logloss")
                m.fit(Xtr, y[tr])
                p = m.predict_proba(Xte)[:, 1]
                scores.append(float(roc_auc_score(y[te], p)))
        results.append((np.mean(scores), params))

    results.sort(reverse=not regression)  # higher AUC better; lower RMSE better
    best = results[0] if not regression else results[0]
    print(f"target={args.target} features={args.features} folds={args.n_folds}")
    print("ranked (best first):")
    for score, params in results:
        metric = "auc" if not regression else "rmse"
        print(f"  {metric}={score:.4f}  {params}")
    print(f"\nBEST: {best[1]}")


if __name__ == "__main__":
    main()
