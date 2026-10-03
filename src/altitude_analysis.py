"""Altitude effect on OUTCOMES (raw data, no model).

Connects the physics finding (less movement at altitude) to actual outcomes:
does altitude change whiff/chase/called-strike/hard-contact/HR rates, after
controlling for pitch type?  This is the H1-H4 chain: movement -> outcome ->
run value.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def main():
    df = pd.read_parquet("stuff_model_df.parquet")
    df["EffectiveVelo"] = pd.to_numeric(df["EffectiveVelo"], errors="coerce")

    swing = df[df["is_swing"] == 1]
    take = df[df["is_swing"] == 0]
    batted = df[df["is_batted"] == 1]

    # outcome rates
    swing = swing.copy()
    swing["chase"] = (swing["swung_outside_strike_zone"] == 1).astype(int)
    swing["whiff"] = (swing["is_whiff"] == 1).astype(int)
    take = take.copy()
    take["called_strike"] = (take["is_called_strike"] == 1).astype(int)
    batted = batted.copy()
    batted["hard"] = (batted["ExitSpeed"] >= 95).astype(int)
    batted["groundball"] = (batted["hit_type"] == "GroundBall").astype(int)
    batted["homerun"] = (batted["home_run"] == 1).astype(int)

    def rates(sub, cols, by="altitude_category"):
        return sub.groupby(by, observed=True)[cols].mean()

    print("=== WHIFF RATE by altitude (among swings) ===")
    print(swing.groupby("altitude_category", observed=True)["whiff"].agg(["mean", "count"]).round(4).to_string())
    print("\n=== CHASE RATE by altitude ===")
    print(swing.groupby("altitude_category", observed=True)["chase"].agg(["mean", "count"]).round(4).to_string())
    print("\n=== CALLED STRIKE RATE by altitude (among takes) ===")
    print(take.groupby("altitude_category", observed=True)["called_strike"].agg(["mean", "count"]).round(4).to_string())
    print("\n=== HARD CONTACT + GB + HR by altitude (among batted) ===")
    print(batted.groupby("altitude_category", observed=True)[["hard", "groundball", "homerun"]].mean().round(4).to_string())
    print("\n=== EXIT VELOCITY mean by altitude ===")
    print(batted.groupby("altitude_category", observed=True)["ExitSpeed"].mean().round(1).to_string())

    # whiff by pitch type x altitude (the H1/H2 mechanism)
    print("\n=== WHIFF RATE by pitch type x altitude ===")
    piv = swing.pivot_table(index="AutoPitchType", columns="altitude_category",
                            values="whiff", aggfunc="mean", observed=True).round(3)
    piv["Extreme-NoAlt"] = piv["Extreme Altitude"] - piv["No Altitude"]
    print(piv.to_string())

    print("\n=== HARD CONTACT RATE by pitch type x altitude ===")
    piv2 = batted.pivot_table(index="AutoPitchType", columns="altitude_category",
                              values="hard", aggfunc="mean", observed=True).round(3)
    piv2["Extreme-NoAlt"] = piv2["Extreme Altitude"] - piv2["No Altitude"]
    print(piv2.to_string())


if __name__ == "__main__":
    main()
