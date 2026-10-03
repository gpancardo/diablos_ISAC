"""Sabermetric reliability analysis — the credibility test a PhD would demand.

1. Year-over-year stability of whiff rate (actual): is whiff a repeatable skill?
2. Does the model's Stuff+ in year T predict the ACTUAL whiff rate in T+1?
   (predictive validity — does the metric measure a real, stable skill?)
3. Empirical-Bayes shrinkage: how much to regress a pitcher's observed whiff
   toward the league mean (the "consistency" deliverable).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def main():
    oof = pd.read_parquet("results/r2__whiff__xgb__stuff__pitcher.oof.parquet")

    # per pitcher-year
    py = oof.groupby(["pitcher", "year"], observed=True).agg(
        n=("y", "size"),
        actual_whiff=("y", "mean"),
        pred_whiff=("pred", "mean"),
    ).reset_index()

    print("=== 1. Year-over-year stability (actual whiff rate) ===")
    for y1, y2 in [(2024, 2025), (2025, 2026)]:
        a = py[py.year == y1].set_index("pitcher")
        b = py[py.year == y2].set_index("pitcher")
        both = a.join(b, lsuffix="_1", rsuffix="_2")
        both = both[(both.n_1 >= 50) & (both.n_2 >= 50)]
        if len(both) < 10:
            print(f"  {y1}->{y2}: insufficient overlap (n={len(both)})")
            continue
        r_actual = both.actual_whiff_1.corr(both.actual_whiff_2)
        r_pred_actual = both.pred_whiff_1.corr(both.actual_whiff_2)
        r_pred_pred = both.pred_whiff_1.corr(both.pred_whiff_2)
        print(f"  {y1}->{y2} (n={len(both)}):")
        print(f"    actual vs actual  r={r_actual:.3f}")
        print(f"    model(T) vs actual(T+1)  r={r_pred_actual:.3f}  <- predictive validity")
        print(f"    model vs model    r={r_pred_pred:.3f}")

    print("\n=== 2. Empirical-Bayes shrinkage (regression to the mean) ===")
    # variance decomposition: within-pitcher vs between-pitcher
    g = oof.groupby("pitcher", observed=True)["y"]
    n_pitcher = g.size()
    mean_pitcher = g.mean()
    grand_mean = oof["y"].mean()
    between_var = mean_pitcher.var()          # true skill variance (unshrunk, biased up)
    within_var = oof["y"].var() - between_var  # binomial-ish noise
    within_var = max(within_var, 1e-6)

    # shrinkage factor per pitcher: B = between / (between + within/n)
    shrink = between_var / (between_var + within_var / n_pitcher)
    shrunk = grand_mean + shrink * (mean_pitcher - grand_mean)

    res = pd.DataFrame({
        "n": n_pitcher, "observed_whiff": mean_pitcher, "shrunk_whiff": shrunk,
    }).sort_values("shrunk_whiff", ascending=False)
    print(f"grand mean whiff={grand_mean:.4f}  between_var={between_var:.5f}  "
          f"within_var={within_var:.5f}")
    print(f"median shrinkage factor={shrink.median():.3f} (1.0 = no shrink, 0 = all league mean)")
    print("\ntop 10 pitchers by SHRUNK whiff ability (adjusted for sample size):")
    print(res.head(10).round(4).to_string())

    res.to_csv("results/pitcher_shrunk_whiff.csv")
    print("\nsaved results/pitcher_shrunk_whiff.csv")


if __name__ == "__main__":
    main()
