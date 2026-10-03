"""Feature engineering for the Stuff+ model.

Leakage discipline (the hard requirement of the competition):

  1. FEATURES are built ONLY from columns the data dictionary marks
     stuff_feature / location_plus_only / context_only.  All target_only,
     grouping_only and id_only columns are excluded by construction.

  2. ARSENAL features (differentials vs the pitcher's fastball, pitch usage,
     tunnel similarity) are pitcher-level aggregates.  They are computed from a
     `baseline_df` (the training fold) and then applied to any other frame, so a
     held-out pitcher or season NEVER contributes its own statistics to its own
     features.  Pitchers absent from the baseline (cold start, e.g. the pitcher
     holdout) fall back to the league-median prior.

  3. Scaling/imputation/encoding are fit on the training fold only.

Feature sets:
  stuff       release-point physics only            -> Stuff+ sub-model
  location    plate location + strike-zone flags    -> Location+ sub-model
  context     count / inning / outs / half          -> Pitching+ context
  altitude    altitude_category                     -> environment
  arsenal     differentials vs fastball + usage     -> arsenal-as-context
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- raw feature columns (release-point physics) --------------------------
# Tilt and ZoneSpeed are dropped (redundant with SpinAxis / RelSpeed, corr>0.98).
# ZoneSpeed/EffectiveVelo/ZoneTime are all ~collinear with RelSpeed; kept the
# physically distinct ones only.
STUFF_FEATURES = [
    "RelSpeed", "EffectiveVelo", "SpinRate", "SpinAxis",
    "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak", "VertBreak",
    "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle",
    "SpeedDrop", "pfxx", "pfxz",
]

# De-correlated stuff set (Sept 2026 fix): removes the near-duplicate columns
# that were splitting SHAP importance and adding nothing to AUC.
#   EffectiveVelo  corr 0.985 with RelSpeed
#   VertBreak      corr 0.947 with InducedVertBreak (and is IVB minus gravity,
#                  i.e. a deterministic function of IVB + flight time)
#   pfxx           corr 0.995 with HorzBreak
#   pfxz           corr 0.997 with InducedVertBreak
# Keeping one representative of each collinear group makes the tree's feature
# importances (and hence the SHAP "physical profile") interpretable.
STUFF_FEATURES_CLEAN = [
    "RelSpeed", "SpinRate", "SpinAxis",
    "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak",
    "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle",
    "SpeedDrop",
]

LOCATION_FEATURES = [
    "PlateLocHeight", "PlateLocSide", "in_strike_zone", "outside_strike_zone",
]

CONTEXT_FEATURES = ["Inning", "Outs", "Balls", "Strikes"]

# categoricals (one-hot encoded)
CATEGORICAL_FEATURES = [
    "AutoPitchType", "PitcherThrows", "BatterSide",
    "altitude_category", "Top/Bottom",
]

LOCATION_PLUS_FEATURES = LOCATION_FEATURES + CONTEXT_FEATURES

ARSENAL_FEATURES = [
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
    "pitch_usage_pct", "tunnel_similarity",
]

# which pitch types count as "fastball" for the differential baseline
FASTBALL_TYPES = {"Four-Seam", "Sinker"}

# --- set definitions -------------------------------------------------------
# AutoPitchType / handedness are physical (stuff_feature) categoricals.
_HAND_PITCH = ["AutoPitchType", "PitcherThrows", "BatterSide"]

SETS = {
    "stuff": STUFF_FEATURES + _HAND_PITCH,
    "stuff_clean": STUFF_FEATURES_CLEAN + _HAND_PITCH,
    "stuff+arsenal": STUFF_FEATURES + _HAND_PITCH + ARSENAL_FEATURES,
    "stuff+location": STUFF_FEATURES + _HAND_PITCH + LOCATION_FEATURES,
    # Location+ sub-model: ubication value ONLY -- no release-point physics.
    # Conditioned on pitch cluster (AutoPitchType) + count + batter side, per
    # the brief's own Location+ definition. This isolates command/ubication
    # from stuff, mirroring FanGraphs' Stuff+/Location+ split.
    "location_plus": LOCATION_PLUS_FEATURES + _HAND_PITCH + ["Top/Bottom"],
    "pitching+": (STUFF_FEATURES + _HAND_PITCH + LOCATION_FEATURES
                  + CONTEXT_FEATURES + ["altitude_category", "Top/Bottom"]
                  + ARSENAL_FEATURES),
    # pitching+ WITHOUT arsenal (clean ablation: isolates arsenal's contribution)
    "pitching+_noarsenal": (STUFF_FEATURES + _HAND_PITCH + LOCATION_FEATURES
                            + CONTEXT_FEATURES + ["altitude_category", "Top/Bottom"]),
    "full": (STUFF_FEATURES + _HAND_PITCH + LOCATION_FEATURES
             + CONTEXT_FEATURES + ["altitude_category", "Top/Bottom"]
             + ARSENAL_FEATURES),
    # spin_efficiency = total induced movement per unit spin ("shape over label", H4)
    "stuff+spin": STUFF_FEATURES + _HAND_PITCH + ["spin_efficiency"],
    "pitching+spin": (STUFF_FEATURES + _HAND_PITCH + LOCATION_FEATURES
                      + CONTEXT_FEATURES + ["altitude_category", "Top/Bottom"]
                      + ARSENAL_FEATURES + ["spin_efficiency"]),
    # ablation: drop pfxx/pfxz (movement ~ final location) to test whether
    # "stuff" is secretly encoding plate location (chase/called-strike leak check)
    "stuff_nopfx": ([c for c in STUFF_FEATURES if c not in ("pfxx", "pfxz")]
                    + _HAND_PITCH),
    # small-gains sweep (Oct 2026): clean stuff set + VAA/HAA flatness
    # normalized by velocity + mirror-symmetric |HorzBreak|.
    "stuff_v2": (STUFF_FEATURES_CLEAN + _HAND_PITCH
                + ["vaa_per_velo", "haa_per_velo", "horz_break_abs"]),
}


def _fill_rare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN/Undefined categoricals with a sentinel so nothing is dropped."""
    out = df.copy()
    for c in ["PitcherThrows", "BatterSide", "altitude_category", "Top/Bottom"]:
        if c in out.columns:
            out[c] = out[c].fillna("Unknown").astype(str).replace("", "Unknown")
    if "AutoPitchType" in out.columns:
        out["AutoPitchType"] = out["AutoPitchType"].fillna("Unknown").astype(str)
    return out


def _fastball_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher fastball baselines (velocity, movement, release point)."""
    fb = df[df["AutoPitchType"].isin(FASTBALL_TYPES)]
    g = fb.groupby("pitcher_anon_id", observed=True).agg(
        fb_velo=("RelSpeed", "mean"),
        fb_ivb=("InducedVertBreak", "mean"),
        fb_hb=("HorzBreak", "mean"),
        fb_relh=("RelHeight", "mean"),
        fb_rels=("RelSide", "mean"),
        fb_ext=("Extension", "mean"),
    )
    return g


def _pitch_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher pitch-type usage fraction (over all their pitches)."""
    n_total = df.groupby("pitcher_anon_id", observed=True).size().rename("n_total")
    usage = (
        df.groupby(["pitcher_anon_id", "AutoPitchType"], observed=True)
        .size()
        .rename("n_type")
        .reset_index()
    )
    usage = usage.merge(n_total, on="pitcher_anon_id")
    usage["pitch_usage_pct"] = usage["n_type"] / usage["n_total"]
    return usage[["pitcher_anon_id", "AutoPitchType", "pitch_usage_pct"]]


def _league_priors(baseline: pd.DataFrame) -> dict:
    """Median fastball baselines over the baseline frame (cold-start prior)."""
    fb = baseline[baseline["AutoPitchType"].isin(FASTBALL_TYPES)]
    return {
        "fb_velo": fb["RelSpeed"].median(),
        "fb_ivb": fb["InducedVertBreak"].median(),
        "fb_hb": fb["HorzBreak"].median(),
        "fb_relh": fb["RelHeight"].median(),
        "fb_rels": fb["RelSide"].median(),
        "fb_ext": fb["Extension"].median(),
        "usage": 1.0 / baseline["AutoPitchType"].nunique(),  # flat prior
    }


def build_arsenal(apply_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Add arsenal feature columns to apply_df using baseline_df statistics.

    Any pitcher not present in baseline_df gets the league-median prior.
    """
    fb = _fastball_baselines(baseline_df)
    usage = _pitch_usage(baseline_df)
    prior = _league_priors(baseline_df)

    out = apply_df.copy()
    out = out.merge(fb, on="pitcher_anon_id", how="left")
    out = out.merge(usage, on=["pitcher_anon_id", "AutoPitchType"], how="left")

    # cold-start fallback (pitcher unseen in baseline)
    out["fb_velo"] = out["fb_velo"].fillna(prior["fb_velo"])
    out["fb_ivb"] = out["fb_ivb"].fillna(prior["fb_ivb"])
    out["fb_hb"] = out["fb_hb"].fillna(prior["fb_hb"])
    out["fb_relh"] = out["fb_relh"].fillna(prior["fb_relh"])
    out["fb_rels"] = out["fb_rels"].fillna(prior["fb_rels"])
    out["fb_ext"] = out["fb_ext"].fillna(prior["fb_ext"])
    out["pitch_usage_pct"] = out["pitch_usage_pct"].fillna(prior["usage"])

    out["velo_diff_vs_fb"] = out["RelSpeed"] - out["fb_velo"]
    out["ivb_diff_vs_fb"] = out["InducedVertBreak"] - out["fb_ivb"]
    out["hb_diff_vs_fb"] = out["HorzBreak"] - out["fb_hb"]
    # negative Euclidean distance -> higher = closer release point to the FB
    out["tunnel_similarity"] = -np.sqrt(
        (out["RelHeight"] - out["fb_relh"]) ** 2
        + (out["RelSide"] - out["fb_rels"]) ** 2
        + (out["Extension"] - out["fb_ext"]) ** 2
    )
    return out


def fit_categorical_levels(df: pd.DataFrame) -> dict:
    """Return the fixed category vocabulary for one-hot encoding.

    Derived once from the full dataset.  Categories are a fixed ontology
    (pitch types, handedness, altitude buckets), NOT target values, so using
    the full vocabulary introduces no leakage -- it only ensures train/test
    share an identical one-hot column space.
    """
    df = _fill_rare_categoricals(df)
    return {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES}


def build_X(apply_df: pd.DataFrame, baseline_df: pd.DataFrame, set_name: str,
            cat_levels: dict | None = None):
    """Return (X DataFrame, feature_names) for the requested feature set.

    arsenal features (if requested) are computed fold-aware from baseline_df.
    cat_levels fixes the one-hot column space across folds.
    """
    cols = SETS[set_name]
    need_arsenal = any(c in cols for c in ARSENAL_FEATURES)

    df = _fill_rare_categoricals(apply_df)
    if need_arsenal:
        df = build_arsenal(df, baseline_df)

    num_cols = [c for c in cols if c not in CATEGORICAL_FEATURES]
    cat_cols = [c for c in cols if c in CATEGORICAL_FEATURES]

    # engineered feature: induced movement per unit spin (H4 "shape over label")
    if "spin_efficiency" in num_cols:
        df = df.copy()
        df["spin_efficiency"] = np.sqrt(
            df["InducedVertBreak"] ** 2 + df["HorzBreak"] ** 2
        ) / (df["SpinRate"] / 100.0)

    # engineered features (Oct 2026 small-gains sweep):
    #   vaa_per_velo / haa_per_velo -- approach-angle "flatness" normalized by
    #   velocity (a flat VAA is more deceptive at high velo than at low velo;
    #   literature: Driveline "flat approach angle" premium).
    #   horz_break_abs -- mirror-symmetric break magnitude (HorzBreak's sign
    #   flips with pitcher handedness; the signed value forces the tree to
    #   re-learn the mirror split that PitcherThrows already encodes).
    if "vaa_per_velo" in num_cols:
        df = df.copy()
        df["vaa_per_velo"] = df["VertApprAngle"] / df["RelSpeed"]
    if "haa_per_velo" in num_cols:
        df = df.copy()
        df["haa_per_velo"] = df["HorzApprAngle"] / df["RelSpeed"]
    if "horz_break_abs" in num_cols:
        df = df.copy()
        df["horz_break_abs"] = df["HorzBreak"].abs()

    X = df[num_cols].astype(float).copy()
    for c in cat_cols:
        cats = (cat_levels or {}).get(c)
        # wrap in Series to preserve the (non-contiguous) fold index -- a bare
        # pd.Categorical loses its index, and concat would outer-join on index.
        cat = pd.Series(pd.Categorical(df[c], categories=cats), index=df.index)
        dummies = pd.get_dummies(cat, prefix=c, drop_first=False).astype(float)
        X = pd.concat([X, dummies], axis=1)
    return X, list(X.columns)


def build_features_full(df: pd.DataFrame, baseline_df: pd.DataFrame, set_name: str):
    """Convenience: build_X and also return pitcher/game/year ids for grouping."""
    X, names = build_X(df, baseline_df, set_name)
    return X, names
