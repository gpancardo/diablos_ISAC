"""Aggregate results/*.json into a readable comparison table.

Reads every experiment result and prints one line per run, grouped by target.
"""
from __future__ import annotations

import glob
import json
import os


def main():
    rows = []
    for f in sorted(glob.glob("results/*__*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d["mean"]
        row = {
            "target": d["target"],
            "model": d["model"],
            "features": d["features"],
            "holdout": d["holdout"],
            "folds": d["n_folds"],
        }
        if "rmse" in m:  # regression target
            row["rmse"] = round(m["rmse"], 4)
            row["rmse_ratio"] = round(m["rmse_ratio"], 4)
            row["mae"] = round(m["mae"], 4)
        else:
            row["auc"] = round(m["auc"], 4)
            row["brier"] = round(m["brier"], 4)
            row["ece"] = round(m.get("ece", 0), 4)
            row["null_auc"] = round(m.get("null_auc", 0), 4)
        rows.append(row)

    # sort: regression first, then classification by target
    rows.sort(key=lambda r: (r["target"] != "xrun_value", r["target"], r["model"]))

    hdr = ["target", "model", "features", "holdout", "folds"]
    if any(r["target"] == "xrun_value" for r in rows):
        hdr += ["rmse", "rmse_ratio", "mae"]
    hdr += ["auc", "brier", "null_auc"]
    print(" | ".join(hdr))
    print("-" * len(" | ".join(hdr)))
    for r in rows:
        cells = [str(r.get(h, "") if r.get(h, "") != "" else "") for h in hdr]
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
