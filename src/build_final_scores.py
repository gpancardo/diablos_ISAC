"""Build the three-layer deliverable: Stuff+, Location+, Pitching+/Out+ per pitch.

Mirrors the brief's own decomposition (FanGraphs Stuff+/Location+/Pitching+):

  Stuff+      whiff model, release-point physics only (classification scale,
              100+10*z(P(whiff)) within pitch type x season)
  Location+   xrun_value_re model, location+count+batter-side only, NO physics
              (run-value scale, inverted: lower run value allowed = better)
  Pitching+   xrun_value_re model, full feature set (stuff+location+arsenal+
              context+altitude) -- the brief's "combinación final"

Stuff+ and Location+/Pitching+ live on different underlying scales (whiff
probability vs. run value) by design -- each is independently centered at 100
within pitch type x season, exactly like FanGraphs reports them as separate
columns, not a single number. This script just joins them per PitchUID.

Run after the three training commands:
  python src/train.py --target whiff         --model xgb      --features stuff          --holdout pitcher --tag r2     (already done)
  python src/train.py --target xrun_value_re --model ensemble --features pitching+       --holdout pitcher --tag q      (already done)
  python src/train.py --target xrun_value_re --model xgb      --features location_plus   --holdout pitcher --tag locplus
"""
from __future__ import annotations

import pandas as pd

from stuff_score import load_oof, to_stuff_plus


def main():
    stuff = load_oof("results/r2__whiff__xgb__stuff__pitcher.oof.parquet")
    stuff = to_stuff_plus(stuff, col="stuff_plus")[["PitchUID", "stuff_plus"]]

    location = load_oof("results/locplus__xrun_value_re__xgb__location_plus__pitcher.oof.parquet")
    location = to_stuff_plus(location, invert=True, col="location_plus")[["PitchUID", "location_plus"]]

    pitching = load_oof("results/q__xrun_value_re__ensemble__pitching+__pitcher.oof.parquet")
    pitching = to_stuff_plus(pitching, invert=True, col="pitching_plus")
    meta = pitching[["PitchUID", "pitcher", "pitch_type", "altitude", "year", "balls", "strikes"]]
    pitching = pitching[["PitchUID", "pitching_plus"]]

    # outer-merge: whiff/Stuff+ only exists for swings; Location+/Pitching+
    # exist for every pitch. Missing Stuff+ (non-swing pitches) is left NaN,
    # not imputed -- it genuinely has no whiff-based physics score.
    out = meta.merge(stuff, on="PitchUID", how="left")
    out = out.merge(location, on="PitchUID", how="left")
    out = out.merge(pitching, on="PitchUID", how="left")

    out.to_parquet("results/final_scores_pitch.parquet", index=False)
    print(f"wrote results/final_scores_pitch.parquet ({len(out)} pitches, "
          f"{out['stuff_plus'].notna().sum()} with Stuff+)")

    by_pitcher = out.groupby("pitcher", observed=True).agg(
        stuff_plus=("stuff_plus", "mean"),
        location_plus=("location_plus", "mean"),
        pitching_plus=("pitching_plus", "mean"),
        n_pitches=("pitching_plus", "size"),
    ).sort_values("pitching_plus", ascending=False)
    by_pitcher.to_csv("results/final_scores_by_pitcher.csv")
    by_pitcher.to_csv("dashboard/data/final_scores_by_pitcher.csv")  # dashboard copy

    by_arsenal = out.groupby(["pitch_type", "altitude"], observed=True).agg(
        stuff_plus=("stuff_plus", "mean"),
        location_plus=("location_plus", "mean"),
        pitching_plus=("pitching_plus", "mean"),
        n=("pitching_plus", "size"),
    )
    by_arsenal.to_csv("results/final_scores_by_arsenal.csv")

    print("\n--- top 10 pitchers by Pitching+ (n>=200) ---")
    print(by_pitcher[by_pitcher.n_pitches >= 200].head(10).round(2).to_string())
    print("\n--- mean by pitch type x altitude ---")
    print(by_arsenal.round(2).to_string())

    # dashboard asset: long-format delta (sea level vs extreme altitude) for
    # all three layers, for the "Stuff+ vs Location+ vs Pitching+" comparison
    # chart (entregable 4: comparación lado a lado).
    piv = by_arsenal.reset_index()
    rows = []
    for layer in ["stuff_plus", "location_plus", "pitching_plus"]:
        sea = piv[piv.altitude == "No Altitude"].set_index("pitch_type")[layer]
        ext = piv[piv.altitude == "Extreme Altitude"].set_index("pitch_type")[layer]
        for pt in sorted(set(sea.index) & set(ext.index)):
            rows.append({"pitch_type": pt, "layer": layer,
                        "plus_sea": sea[pt], "plus_extreme": ext[pt],
                        "delta": ext[pt] - sea[pt]})
    pd.DataFrame(rows).to_csv("dashboard/data/layer_altitude_delta.csv", index=False)
    print("\nwrote dashboard/data/layer_altitude_delta.csv")


if __name__ == "__main__":
    main()
