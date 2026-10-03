"""Quantify the altitude effect on pitch value via a counterfactual.

Trains a pitching+ xrun_value_re model, then for a sample of pitches predicts
the SAME pitch under Extreme Altitude vs No Altitude (swapping only the one-hot
altitude columns).  The delta = Stuff+ loss/gain from moving a pitch shape to
the Harp Helú environment.  This is the explicit H1-H4 quantification.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import features
import targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=100000)
    ap.add_argument("--out", default="results/altitude_delta")
    args = ap.parse_args()

    df = targets.build_targets(targets.load())
    cat_levels = features.fit_categorical_levels(df)

    # train on everything but use xrun_value_re
    X, names = features.build_X(df, df, "pitching+", cat_levels)
    y = df["xrun_value_re"].values.astype(float)

    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         tree_method="hist", n_jobs=4, random_state=0, verbosity=0)
    model.fit(X, y)

    # counterfactual sample (hold out a subset, swap altitude)
    samp = df.sample(min(args.n_sample, len(df)), random_state=0)
    Xs, _ = features.build_X(samp, df, "pitching+", cat_levels)
    alt_cols = [c for c in Xs.columns if c.startswith("altitude_category_")]
    extreme_col = "altitude_category_Extreme Altitude"
    noalt_col = "altitude_category_No Altitude"

    X_ext = Xs.copy()
    X_sea = Xs.copy()
    # force extreme / no-altitude one-hots (drop Unknown/Medium)
    for c in alt_cols:
        X_ext[c] = 0.0
        X_sea[c] = 0.0
    X_ext[extreme_col] = 1.0
    X_sea[noalt_col] = 1.0

    pred_ext = model.predict(X_ext)
    pred_sea = model.predict(X_sea)
    delta = pred_ext - pred_sea  # runs lost (negative) / gained at altitude

    out = samp[["pitcher_anon_id", "AutoPitchType", "RelSpeed",
                "InducedVertBreak", "HorzBreak", "SpinRate"]].copy()
    out["delta_rv_altitude"] = delta
    out["pred_sea"] = pred_sea
    out["pred_extreme"] = pred_ext

    # aggregate delta by pitch type
    agg = out.groupby("AutoPitchType", observed=True).agg(
        mean_delta=("delta_rv_altitude", "mean"),
        pct_negative=("delta_rv_altitude", lambda s: (s < 0).mean()),
        n=("delta_rv_altitude", "size"),
    ).sort_values("mean_delta")
    print("--- mean run-value delta moving pitch to Extreme Altitude (negative = worse) ---")
    print(agg.round(4).to_string())
    out.to_parquet(f"{args.out}.parquet", index=False)
    agg.to_csv(f"{args.out}_by_pitchtype.csv")


if __name__ == "__main__":
    main()
