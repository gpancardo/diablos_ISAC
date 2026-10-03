"""Build the Stuff+ score from out-of-fold predictions.

Stuff+ = 100 + 10 · z(pred), normalized WITHIN pitch type (and season), so a
slider is judged relative to other sliders, a fastball relative to other
fastballs (the canonical FanGraphs normalization).

Input: an OOF parquet produced by train.py (columns: PitchUID, pred, pitch_type,
altitude, year, pitcher).  Outputs pitch-level scores plus pitcher/arsenal
aggregations.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_oof(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern))
    assert files, f"no OOF files match {glob_pattern}"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def to_stuff_plus(oof: pd.DataFrame, group_cols=("pitch_type", "year"),
                   invert: bool = False, col: str = "stuff_plus") -> pd.DataFrame:
    """Normalize pred -> Plus score within each group (pitch type x season).

    invert=True for targets where a LOWER prediction is better for the pitcher
    (e.g. run value allowed): the sign is flipped before z-scoring so 100+ always
    means "better than league average", matching ERA+/OPS+ convention.
    """
    oof = oof.copy()
    pred = -oof["pred"] if invert else oof["pred"]
    g = pred.groupby([oof[c] for c in group_cols], observed=True)
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    oof[col] = 100 + 10 * (pred - mean) / std
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="glob of OOF parquet files")
    ap.add_argument("--out", default="results/stuff_plus")
    ap.add_argument("--invert", action="store_true",
                    help="lower pred = better for the pitcher (e.g. run value)")
    ap.add_argument("--col", default="stuff_plus", help="name of the output Plus column")
    args = ap.parse_args()

    oof = load_oof(args.oof)
    oof = to_stuff_plus(oof, invert=args.invert, col=args.col)
    col = args.col

    # pitch-level scores
    oof.to_parquet(f"{args.out}_pitch.parquet", index=False)

    # pitcher-level aggregation (mean Stuff+, n pitches)
    by_pitcher = oof.groupby("pitcher", observed=True).agg(
        **{col: (col, "mean")}, n_pitches=(col, "size"),
    ).sort_values(col, ascending=False)
    by_pitcher.to_csv(f"{args.out}_by_pitcher.csv")
    print(f"--- top 15 pitchers by {col} (n>=200) ---")
    print(by_pitcher[by_pitcher.n_pitches >= 200].head(15).round(2).to_string())

    # arsenal (pitch type) aggregation, and altitude split
    by_arsenal = oof.groupby(["pitch_type", "altitude"], observed=True).agg(
        **{col: (col, "mean")}, n=(col, "size"),
    )
    print(f"\n--- {col} by pitch type x altitude ---")
    print(by_arsenal.round(2).to_string())
    by_arsenal.to_csv(f"{args.out}_by_arsenal.csv")

    # altitude effect (the H1-H4 quantification)
    alt = oof.groupby("altitude", observed=True)[col].agg(["mean", "count"])
    print("\n--- mean Stuff+ by altitude (should be ~100 each; delta shows raw pred shift) ---")
    print(alt.round(2).to_string())


if __name__ == "__main__":
    main()
