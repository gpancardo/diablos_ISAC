"""Quality EDA for Diablos Rojos Stuff+ hackathon.

Loads the pitch-level parquet, profiles schema/missing/cardinality,
audits leakage (feature_role), and quantifies the altitude effect.
Writes a markdown report to results/eda_report.md and prints highlights.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA = "stuff_model_df.parquet"
DICT = "stuff_plus_data_dictionary.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA)
dd = pd.read_csv(DICT)

# feature-role map from the data dictionary (authoritative leakage source)
role = dict(zip(dd["column_name"], dd["feature_role"]))
use_feat = dict(zip(dd["column_name"], dd["use_as_model_feature"]))

print("=" * 78)
print("SHAPE", df.shape)
print("=" * 78)

# --- EffectiveVelo is a string -> convert ---
def to_num(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------
missing = df.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]

# ---------------------------------------------------------------------------
# Cardinality / duplicates
# ---------------------------------------------------------------------------
n_pitchuid = df["PitchUID"].nunique()
n_rows = len(df)
dup = n_rows - n_pitchuid

# ---------------------------------------------------------------------------
# Categorical value counts
# ---------------------------------------------------------------------------
cats = {
    "AutoPitchType": df["AutoPitchType"],
    "PitchCall": df["PitchCall"],
    "play_result": df["play_result"],
    "hit_type": df["hit_type"],
    "altitude_category": df["altitude_category"],
    "PitcherThrows": df["PitcherThrows"],
    "BatterSide": df["BatterSide"],
    "KorBB": df["KorBB"],
    "count": df["count"],
    "Top/Bottom": df["Top/Bottom"],
}

print("\n--- IDENTIFIER CARDINALITY ---")
print(f"rows={n_rows}  PitchUID_unique={n_pitchuid}  dup_uids={dup}")
print(f"pitchers={df['pitcher_anon_id'].nunique()}  "
      f"batters={df['batter_anon_id'].nunique()}  "
      f"catchers={df['catcher_anon_id'].nunique()}  "
      f"games={df['game_anon_id'].nunique()}")

print("\n--- YEAR ---")
print(df["year"].value_counts().sort_index().to_string())

print("\n--- CATEGORICAL DISTRIBUTIONS ---")
for name, s in cats.items():
    print(f"\n[{name}] n_unique={s.nunique()}")
    print(s.value_counts(dropna=False).head(15).to_string())

# ---------------------------------------------------------------------------
# EffectiveVelo string issue
# ---------------------------------------------------------------------------
print("\n--- EffectiveVelo (stored as string) ---")
print("sample raw values:", df["EffectiveVelo"].head(10).tolist())
ev_num = to_num(df["EffectiveVelo"])
print(f"convertible: {ev_num.notna().mean():.3f}  range={ev_num.min():.1f}..{ev_num.max():.1f}")
print("non-numeric examples:", df.loc[ev_num.isna(), "EffectiveVelo"].unique()[:10].tolist())

# ---------------------------------------------------------------------------
# Numeric summary of physical features
# ---------------------------------------------------------------------------
phys = ["RelSpeed", "ZoneSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide",
        "Extension", "VertBreak", "InducedVertBreak", "HorzBreak",
        "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle",
        "SpeedDrop", "ZoneTime", "pfxx", "pfxz",
        "PlateLocHeight", "PlateLocSide", "ExitSpeed", "Angle", "Distance"]
print("\n--- NUMERIC SUMMARY (physical + batted-ball) ---")
print(df[phys].describe().T.round(2).to_string())

# ---------------------------------------------------------------------------
# Missingness report
# ---------------------------------------------------------------------------
print("\n--- MISSING (>0) ---")
if len(missing):
    print((missing * 100).round(2).to_string())
else:
    print("none")

# ---------------------------------------------------------------------------
# Target flag distributions (target_only columns)
# ---------------------------------------------------------------------------
flags = ["is_swing", "is_whiff", "is_contact", "is_called_strike",
         "is_swinging_strike", "is_ball_in_play", "is_batted", "is_hit",
         "single", "double", "triple", "home_run", "base_on_balls",
         "strikeout", "is_hit_by_pitch", "swung_outside_strike_zone"]
print("\n--- TARGET FLAG MEANS (outcome base rates) ---")
print(df[flags].mean().round(4).to_string())

# ---------------------------------------------------------------------------
# Altitude effect (raw, before controlling)
# ---------------------------------------------------------------------------
print("\n--- ALTITUDE CATEGORY x PHYSICS ---")
if df["altitude_category"].nunique() > 1:
    g = df.groupby("altitude_category", observed=True)[
        ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
         "VertBreak", "Extension", "ExitSpeed", "Distance"]
    ].agg(["count", "mean"]).round(2)
    print(g.to_string())

# movement by pitch type x altitude (the core H1 hypothesis)
print("\n--- InducedVertBreak + HorzBreak by pitch type x altitude ---")
pivot = df.pivot_table(
    index="AutoPitchType", columns="altitude_category",
    values=["InducedVertBreak", "HorzBreak", "RelSpeed"],
    aggfunc="mean", observed=True,
).round(1)
print(pivot.to_string())

# ---------------------------------------------------------------------------
# Correlation of physical features (Stuff+ candidates)
# ---------------------------------------------------------------------------
corr_cols = ["RelSpeed", "EffectiveVelo", "ZoneSpeed", "SpinRate", "SpinAxis",
             "RelHeight", "RelSide", "Extension", "VertBreak", "InducedVertBreak",
             "HorzBreak", "VertRelAngle", "HorzRelAngle", "VertApprAngle",
             "HorzApprAngle", "SpeedDrop", "ZoneTime", "pfxx", "pfxz"]
tmp = df[corr_cols].copy()
tmp["EffectiveVelo"] = to_num(tmp["EffectiveVelo"])
corr = tmp.corr()
print("\n--- TOP |corr| with InducedVertBreak ---")
print(corr["InducedVertBreak"].abs().sort_values(ascending=False).head(12).round(3).to_string())
print("\n--- TOP |corr| with RelSpeed ---")
print(corr["RelSpeed"].abs().sort_values(ascending=False).head(12).round(3).to_string())

# ---------------------------------------------------------------------------
# Leakage audit: classify every column by feature_role
# ---------------------------------------------------------------------------
print("\n--- LEAKAGE AUDIT (feature_role) ---")
for r in ["stuff_feature", "location_plus_only", "context_only",
          "grouping_only", "id_only", "target_only"]:
    cols = [c for c, v in role.items() if v == r]
    print(f"\n{r} ({len(cols)}):")
    print("  " + ", ".join(cols))

# target columns that are numeric and could be accidentally leaked
print("\n--- CHECK: any target_only column in a 'use_as_model_feature=Yes' set? ---")
leak = [c for c in dd.loc[dd["feature_role"] == "target_only", "column_name"]
        if use_feat.get(c, "").startswith("Yes")]
print("potential dictionary inconsistency:", leak if leak else "none")

# save a compact machine-readable profile for the plan step
df[["year", "pitcher_anon_id", "game_anon_id"]].nunique().to_json("results/cardinality.json")

print("\nDONE EDA")
