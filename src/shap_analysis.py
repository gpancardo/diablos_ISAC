"""SHAP feature importance for the Stuff+ outcome models.

Trains a whiff model on physics-only (stuff) features and reports the features
that drive swing-and-miss -- the interpretable "why" for the pitching coach.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import features
import targets


def main():
    df = targets.build_targets(targets.load())
    cat_levels = features.fit_categorical_levels(df)

    swings = df[df["swing"] == 1]
    X, names = features.build_X(swings, swings, "stuff", cat_levels)
    y = swings["y_whiff"].values.astype(float)

    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=400, learning_rate=0.06, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", n_jobs=4, random_state=0,
                          verbosity=0, eval_metric="logloss")
    model.fit(X, y)

    import shap
    # subsample for SHAP (TreeExplainer is O(n·trees))
    idx = np.random.RandomState(0).choice(len(X), size=min(5000, len(X)), replace=False)
    Xs = X.iloc[idx]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    if isinstance(sv, list):  # binary -> take positive class
        sv = sv[1]
    sv = np.asarray(sv)

    # mean |SHAP| per feature (overall importance)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    print("--- top 20 |SHAP| for P(whiff) ---")
    print(imp.head(20).round(5).to_string())
    imp.to_csv("results/shap_whiff_importance.csv")

    # directionality for top numeric features
    print("\n--- direction (mean SHAP for numeric top features) ---")
    for c in imp.index[:12]:
        if c in Xs.columns and pd.api.types.is_numeric_dtype(Xs[c]):
            direction = "increases whiff" if sv[:, names.index(c)].mean() > 0 else "reduces whiff"
            print(f"{c:22s} {direction}")


if __name__ == "__main__":
    main()
