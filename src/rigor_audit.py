"""Rigor audit — close the 3 most-likely judge attacks from the self-critique.

  A. Honest null baselines      -- physics value-add over pitch-type (and +pitcher)
  B. Altitude Whiff+ delta CI   -- is the Four-Seam -3.7 significant?
  C. pfxx/pfxz ablation         -- is "stuff" secretly encoding location?

All leak-free (GroupKFold pitcher; baselines use train-only rates).
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import features
import targets
import train


# ---------------------------------------------------------------------------
# A. honest null baselines for whiff (rate lookups, no ML)
# ---------------------------------------------------------------------------
def null_auc_whiff():
    df = targets.build_targets(targets.load())
    sub = df[targets.TARGET_MASK["whiff"](df)].reset_index(drop=True)
    y = sub["y_whiff"].astype(float).values
    groups = pd.factorize(sub["pitcher_anon_id"])[0]
    pt = sub["AutoPitchType"].values
    pitcher = sub["pitcher_anon_id"].values

    gkf = GroupKFold(n_splits=10)
    res = {"pitch_type_only": [], "pitch_type+pitcher": []}
    for tr, te in gkf.split(np.arange(len(sub)), groups=groups):
        tr_df = pd.DataFrame({"pt": pt[tr], "pitcher": pitcher[tr], "y": y[tr]})
        # pitch type only
        rate_pt = tr_df.groupby("pt", observed=True)["y"].mean()
        p_pt = pd.Series(pt[te]).map(rate_pt).fillna(rate_pt.mean()).values
        res["pitch_type_only"].append(roc_auc_score(y[te], p_pt))
        # pitch type + pitcher (league-mean prior for cold-start pitchers)
        rate_pp = tr_df.groupby(["pitcher", "pt"], observed=True)["y"].mean()
        key = pd.DataFrame({"pitcher": pitcher[te], "pt": pt[te]})
        p_pp = key.merge(rate_pp.rename("r"), on=["pitcher", "pt"], how="left")["r"]
        p_pp = p_pp.fillna(rate_pt.mean()).values
        res["pitch_type+pitcher"].append(roc_auc_score(y[te], p_pp))
    return {k: float(np.mean(v)) for k, v in res.items()}


# ---------------------------------------------------------------------------
# B. bootstrap CI on the Four-Seam Whiff+ altitude delta
# ---------------------------------------------------------------------------
def altitude_delta_ci(oof_path="results/r2__whiff__xgb__stuff__pitcher.oof.parquet",
                      n_boot=500, seed=0):
    oof = pd.read_parquet(oof_path)
    # whiff+ = 100 + 10 * z(pred) within (pitch_type, year)
    z = oof.groupby(["pitch_type", "year"], observed=True)["pred"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    oof["whiff_plus"] = 100 + 10 * z
    oof["four_seam"] = (oof["pitch_type"] == "Four-Seam")

    fs = oof[oof["four_seam"]]
    no_alt = fs[fs["altitude"].isin(["No Altitude"])]
    ext = fs[fs["altitude"].isin(["Extreme Altitude"])]

    def delta_of(d):
        n = d[d["altitude"].isin(["No Altitude"])]["whiff_plus"].mean()
        e = d[d["altitude"].isin(["Extreme Altitude"])]["whiff_plus"].mean()
        return e - n

    point = delta_of(fs)
    # clustered bootstrap by pitcher (resample pitchers within each altitude)
    rng = np.random.RandomState(seed)
    pitchers = fs["pitcher"].unique()
    p_index = {p: np.where(fs["pitcher"].values == p)[0] for p in pitchers}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(pitchers, size=len(pitchers), replace=True)
        idx = np.concatenate([p_index[p] for p in sample])
        deltas[b] = delta_of(fs.iloc[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"four_seam_delta": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "n_no_alt": int(len(no_alt)), "n_extreme": int(len(ext))}


# ---------------------------------------------------------------------------
# C. pfxx/pfxz ablation: whiff / chase / called_strike with vs without pfxx,pfxz
# ---------------------------------------------------------------------------
def run_ablation():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [
        ("audit", "whiff", "xgb", "stuff_nopfx", "pitcher", 10),
        ("audit", "chase", "xgb", "stuff_nopfx", "pitcher", 10),
        ("audit", "called_strike", "xgb", "stuff_nopfx", "pitcher", 10),
    ]
    out = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(train.run_experiment, *j[1:], 2, j[0]): j[2] for j in jobs}
        for fut in as_completed(futs):
            target = futs[fut]
            r = fut.result()
            out[target] = {"auc": r["mean"]["auc"], "std": r["std"]["auc"]}
    return out


if __name__ == "__main__":
    print("=== A. honest null baselines (whiff) ===", flush=True)
    nulls = null_auc_whiff()
    print(json.dumps(nulls, indent=2), flush=True)

    print("=== B. Four-Seam altitude Whiff+ delta CI ===", flush=True)
    ci = altitude_delta_ci()
    print(json.dumps(ci, indent=2), flush=True)

    print("=== C. pfxx/pfxz ablation (AUC stuff vs stuff_nopfx) ===", flush=True)
    ab = run_ablation()
    print(json.dumps(ab, indent=2), flush=True)

    with open("results/rigor_audit.json", "w") as f:
        json.dump({"nulls": nulls, "altitude_ci": ci, "ablation": ab}, f, indent=2)
    print("wrote results/rigor_audit.json", flush=True)
