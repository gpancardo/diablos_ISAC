"""Target construction for the Stuff+ model.

All targets are derived exclusively from *outcome* columns (feature_role =
target_only). No target column is ever used as a model feature, so there is no
outcome leakage by construction.

Targets (per the competition's multi-objective spec):
  xrun_value      (regression, all pitches)  linear-weights run value of the PA
  whiff           (binary, swings only)      swing-and-miss
  chase           (binary, swings only)      swing at pitch outside the zone
  called_strike   (binary, takes only)       called strike
  weak_contact    (binary, batted balls)     ExitSpeed < 80 mph
  groundball      (binary, batted balls)     hit_type == GroundBall
  barrel          (binary, batted balls)     hard+optimal-angle contact (proxy)
  out             (binary, all pitches)      PA ended in an out
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- linear weights (runs above average, per terminal event) ---
# Standard values; non-terminal events carry 0 (their value accrues to the
# terminal event of the PA). No base-out state is available, so we use
# event linear weights rather than RE24. Flagged as a known simplification.
LINEAR_WEIGHTS = {
    "HomeRun": 1.40,
    "Triple": 1.09,
    "Double": 0.77,
    "Single": 0.47,
    "Error": 0.48,
    "Walk": 0.32,
    "HitByPitch": 0.32,
    "Sacrifice": -0.10,
    "FieldersChoice": -0.30,
    "Out": -0.27,
    "Strikeout": -0.30,
    # non-terminal (PA continues) -> 0
    "BallCalled": 0.0,
    "BallinDirt": 0.0,
    "NeutralPlay": 0.0,
    "StrikeSwinging": 0.0,
    "FoulBallFieldable": 0.0,
    "Undefined": 0.0,
}

WEAK_CONTACT_MPH = 80.0      # "weak" exit velocity threshold (mph)
BARREL_EV_MIN = 95.0         # hard-contact exit velocity (mph)
BARREL_ANGLE_LO = 20.0       # barrel launch-angle window (deg)
BARREL_ANGLE_HI = 40.0

# Terminal balls-in-play results (the ONLY correct denominator for batted-ball
# targets).  `is_batted` also flags fouls and even non-contact events, so it is
# NOT used.  Verified: every true ball-in-play has is_batted==1, but is_batted
# additionally marks ~109k foul contacts (NeutralPlay) + 1.1k impossible rows.
BATTED_IN_PLAY_RESULTS = {
    "Single", "Double", "Triple", "HomeRun",
    "Out", "FieldersChoice", "Error", "Sacrifice",
}


def load() -> pd.DataFrame:
    """Load the pitch-level parquet and fix the EffectiveVelo string column."""
    df = pd.read_parquet("stuff_model_df.parquet")
    df["EffectiveVelo"] = pd.to_numeric(df["EffectiveVelo"], errors="coerce")
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add all target columns to df (in place copy) and return it."""
    df = df.copy()

    # --- regression target: run value (linear weights) ---
    df["xrun_value"] = df["play_result"].map(LINEAR_WEIGHTS).astype(float)

    # --- count-aware run value (RE24-style, count instead of base-out) -------
    # RE(count) = expected run value of resolving the PA from that count,
    # estimated as the mean linear-weight over all pitches at that count
    # (terminal carry their weight, non-terminal carry 0).  Non-terminal pitches
    # then get the marginal value of the count transition.  Vectorized.
    re = df.groupby(["Balls", "Strikes"], observed=True)["xrun_value"].mean()
    # 12-state lookup table (balls 0..3, strikes 0..2)
    re_tab = np.zeros((4, 3))
    for (b, s), v in re.items():
        re_tab[int(b), int(s)] = v

    pr = df["play_result"].values
    balls = df["Balls"].values.astype(int)
    strikes = df["Strikes"].values.astype(int)
    is_ball = (pr == "BallCalled") | (pr == "BallinDirt")
    is_strike = (pr == "NeutralPlay") | (pr == "StrikeSwinging")
    non_terminal = is_ball | is_strike

    nb = np.where(is_ball, np.minimum(balls + 1, 3), balls)
    ns = np.where(is_strike, np.minimum(strikes + 1, 2), strikes)

    re_cur = re_tab[balls, strikes]
    re_next = re_tab[nb, ns]
    df["xrun_value_re"] = np.where(non_terminal, re_next - re_cur, df["xrun_value"].values).astype(float)

    # --- classification targets ---
    df["swing"] = (df["is_swing"] == 1).astype(int)
    df["y_whiff"] = (df["is_whiff"] == 1).astype(int)                    # on swings
    df["y_chase"] = (df["swung_outside_strike_zone"] == 1).astype(int)   # on swings
    df["y_called_strike"] = (df["is_called_strike"] == 1).astype(int)    # on takes
    df["y_out"] = (df["OutsOnPlay"] >= 1).astype(int)                    # all pitches

    # batted-ball-only targets.
    # NOTE (data-quality fix, Sept 2026): the raw `is_batted` flag is BROKEN --
    # it marks ~109k foul contacts (play_result == NeutralPlay) and even
    # impossible rows (BallCalled, Walk, HitByPitch) as batted.  The correct
    # denominator for batted-ball outcomes is the set of terminal
    # balls-in-play results, not `is_batted`.  See plans/EDA_REPORT.md.
    batted = df["play_result"].isin(BATTED_IN_PLAY_RESULTS)
    df["y_weak_contact"] = np.where(batted, (df["ExitSpeed"] < WEAK_CONTACT_MPH).astype(float), np.nan)
    df["y_groundball"] = np.where(batted, (df["hit_type"] == "GroundBall").astype(float), np.nan)
    df["y_barrel"] = np.where(
        batted,
        ((df["ExitSpeed"] >= BARREL_EV_MIN)
         & (df["Angle"] >= BARREL_ANGLE_LO)
         & (df["Angle"] <= BARREL_ANGLE_HI)).astype(float),
        np.nan,
    )
    return df


# subset mask for each classification target (None = all rows)
TARGET_MASK = {
    "xrun_value": None,            # regression, all pitches
    "xrun_value_re": None,         # regression, all pitches (count-aware)
    "out": None,
    "whiff": lambda d: d["swing"] == 1,
    "chase": lambda d: d["swing"] == 1,
    "called_strike": lambda d: d["swing"] == 0,
    "weak_contact": lambda d: d["y_weak_contact"].notna(),
    "groundball": lambda d: d["y_groundball"].notna(),
    "barrel": lambda d: d["y_barrel"].notna(),
}

TARGET_COL = {
    "xrun_value": "xrun_value",
    "xrun_value_re": "xrun_value_re",
    "out": "y_out",
    "whiff": "y_whiff",
    "chase": "y_chase",
    "called_strike": "y_called_strike",
    "weak_contact": "y_weak_contact",
    "groundball": "y_groundball",
    "barrel": "y_barrel",
}

REGRESSION_TARGETS = {"xrun_value", "xrun_value_re"}


def make_y(df: pd.DataFrame, target: str):
    """Return (subset_df, y_series) for a given target name."""
    ycol = TARGET_COL[target]
    mask_fn = TARGET_MASK[target]
    sub = df if mask_fn is None else df[mask_fn(df)]
    return sub, sub[ycol].astype(float)
