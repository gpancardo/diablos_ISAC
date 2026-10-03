"""E4 — Is the xgb+lgb ensemble actually better than the best single?

Paired bootstrap: both models' OOF predictions live on the SAME folds (same
pitcher split, same rows), so we can bootstrap the AUC DIFFERENCE at the pitcher
(and game) level and get a CI + p-value on "ensemble > xgb".

Also does the same for the run-value target (rmse difference).

Output: results/phd_ensemble_test.json
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

raw = pd.read_parquet("stuff_model_df.parquet")[["PitchUID", "game_anon_id"]]


def load_pair(xgb_path, lgb_path):
    a = pd.read_parquet(xgb_path)
    b = pd.read_parquet(lgb_path)[["PitchUID", "pred"]].rename(columns={"pred": "pred_lgb"})
    m = a.merge(b, on="PitchUID").merge(raw, on="PitchUID", how="left")
    m["pred_ens"] = 0.5 * (m.pred + m.pred_lgb)
    return m


def paired_boot(m, score, n_boot=500, seed=0):
    codes = pd.factorize(m.pitcher, use_na_sentinel=False)[0]
    n = codes.max() + 1
    counts = np.bincount(codes, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, n, n)
        idx = np.concatenate([np.arange(starts[c], starts[c + 1]) for c in pick])
        s = m.iloc[idx]
        diffs[b] = score(s)
    return diffs


def main():
    out = {}

    # --- whiff ---
    for tag, xp, lp in [
        ("whiff stuff (r2)", "results/r2__whiff__xgb__stuff__pitcher.oof.parquet",
                            "results/r2__whiff__lgb__stuff__pitcher.oof.parquet"),
    ]:
        m = load_pair(xp, lp)
        d = paired_boot(m, lambda s: roc_auc_score(s.y, s.pred_ens) - roc_auc_score(s.y, s.pred))
        lo, hi = np.percentile(d, [2.5, 97.5])
        pval = float((d <= 0).mean())
        out[tag] = {
            "auc_xgb": float(roc_auc_score(m.y, m.pred)),
            "auc_ensemble": float(roc_auc_score(m.y, m.pred_ens)),
            "diff_mean": float(d.mean()),
            "ci95": [float(lo), float(hi)],
            "p(ens<=xgb)": pval,
        }
        print(f"{tag}: xgb {out[tag]['auc_xgb']:.4f}  ensemble {out[tag]['auc_ensemble']:.4f}  "
              f"diff {d.mean():+.4f}  CI95 [{lo:.4f},{hi:.4f}]  p(ens<=xgb)={pval:.3f}")

    # --- run value ---
    r = load_pair("results/r2__xrun_value_re__xgb__pitching+__pitcher.oof.parquet",
                  "results/r2__xrun_value_re__lgb__pitching+__pitcher.oof.parquet")
    d = paired_boot(r, lambda s: np.sqrt(mean_squared_error(s.y, s.pred)) -
                                  np.sqrt(mean_squared_error(s.y, s.pred_ens)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    out["xrun_value_re pitching+"] = {
        "rmse_xgb": float(np.sqrt(mean_squared_error(r.y, r.pred))),
        "rmse_ensemble": float(np.sqrt(mean_squared_error(r.y, r.pred_ens))),
        "diff_mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p(ens<=xgb)": float((d <= 0).mean()),
    }
    print(f"run value: xgb {out['xrun_value_re pitching+']['rmse_xgb']:.5f}  "
          f"ensemble {out['xrun_value_re pitching+']['rmse_ensemble']:.5f}  "
          f"diff {d.mean():+.5f}  CI95 [{lo:.5f},{hi:.5f}]  p(ens<=xgb)={(d <= 0).mean():.3f}")

    json.dump(out, open("results/phd_ensemble_test.json", "w"), indent=2)
    print("\nwrote results/phd_ensemble_test.json")


if __name__ == "__main__":
    main()
