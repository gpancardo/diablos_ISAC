"""E5 — Physics-channel altitude counterfactual for whiff.

Answers the challenge question honestly: how much of the whiff drop at altitude
is EXPLAINED by the measured movement change (the physics channel), vs a
residual "park" effect the movement doesn't capture?

Method (matched, causal-leaning):
  1. Train a whiff model on release physics (stuff_clean).
  2. Estimate the within-pitcher (fixed-effects) altitude deltas of the movement
     features from pitchers who threw in BOTH No-Altitude and Extreme-Altitude
     parks.  Only movement/approach columns change physically (Magnus ~ density);
     release speed/spin/release-point are release properties and are left alone.
  3. For every Extreme-altitude pitch in the paired sample, construct its
     "sea-level twin" by removing the per-type movement delta, and predict P(whiff)
     for both.  The mean difference is the physics-channel altitude effect.

Output: results/phd_altitude_counterfactual.csv  (per pitch type)
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import features
import targets

# movement/approach columns that physically change with air density
MOVE_COLS = ["InducedVertBreak", "HorzBreak", "VertApprAngle", "HorzApprAngle"]


def within_delta(d, col):
    """Fixed-effects (within-pitcher) estimate of the Extreme-No altitude delta."""
    d = d[["pitcher_anon_id", "EXT", col]].dropna()
    g = d.groupby("pitcher_anon_id")[["EXT", col]].transform("mean")
    x = d.EXT - g.EXT
    y = d[col] - g[col]
    return (x @ y) / (x @ x)


def main():
    df = targets.build_targets(targets.load())
    df = df[df.altitude_category.isin(["No Altitude", "Extreme Altitude"])].copy()
    df["EXT"] = (df.altitude_category == "Extreme Altitude").astype(float)

    # train whiff model on stuff_clean (all swings)
    swings = df[df.swing == 1]
    cat_levels = features.fit_categorical_levels(swings)
    X, names = features.build_X(swings, swings, "stuff_clean", cat_levels)
    y = swings["y_whiff"].values.astype(float)
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      tree_method="hist", n_jobs=4, random_state=0, verbosity=0,
                      eval_metric="logloss")
    m.fit(X, y)

    rows = []
    for ptype, grp in swings.groupby("AutoPitchType", observed=True):
        if ptype not in ("Four-Seam", "Sinker", "Changeup", "Curveball", "Slider", "Cutter", "Splitter"):
            continue
        if grp.EXT.nunique() < 2:
            continue
        # within-pitcher deltas for the movement columns
        deltas = {c: within_delta(grp, c) for c in MOVE_COLS}

        # counterfactual: only on Extreme-altitude pitches of this type
        ext = grp[grp.EXT == 1].copy()
        X_ext, _ = features.build_X(ext, swings, "stuff_clean", cat_levels)
        X_sea = X_ext.copy()
        for c in MOVE_COLS:
            if c in X_sea.columns:
                X_sea[c] = X_sea[c] - deltas[c]   # add back the sea-level movement

        p_ext = m.predict_proba(X_ext)[:, 1]
        p_sea = m.predict_proba(X_sea)[:, 1]
        # observational (raw) gap for comparison
        obs = grp[grp.EXT == 0].y_whiff.mean() - ext.y_whiff.mean()

        rows.append({
            "pitch_type": ptype,
            "n_extreme_pitches": int(len(ext)),
            "d_IVB": round(deltas["InducedVertBreak"], 3),
            "d_HB": round(deltas["HorzBreak"], 3),
            "d_VertAppr": round(deltas["VertApprAngle"], 3),
            "d_HorzAppr": round(deltas["HorzApprAngle"], 3),
            "obs_delta_Pwhiff": round(float(obs), 4),                       # raw observational
            "physics_delta_Pwhiff": round(float((p_ext - p_sea).mean()), 4),  # explained by movement
        })

    out = pd.DataFrame(rows).sort_values("physics_delta_Pwhiff")
    out.to_csv("results/phd_altitude_counterfactual.csv", index=False)
    print(out.to_string(index=False))
    print("\n(fisica_delta negativo = el movimiento reducido explica MENOS whiff en altitud)")
    print("interpretacion: si |physics| ~ |obs|, la altitud actua CASI TODA via movimiento;")


if __name__ == "__main__":
    main()
