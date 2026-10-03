"""Round-2 experiment matrix (leak fixes + count-aware run value + OOF saving).

Runs in parallel with 4 workers (model n_jobs=2 each).  Each job writes
results/<tag>__...json AND a .oof.parquet with out-of-fold predictions.
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

import train

# (tag, target, model, features, holdout, n_folds)
MATRIX = [
    # --- Stuff+ outcomes (honest: physics-only, NO location) ---
    ("r2", "whiff",         "xgb", "stuff", "pitcher", 10),
    ("r2", "chase",         "xgb", "stuff", "pitcher", 10),
    ("r2", "called_strike", "xgb", "stuff", "pitcher", 10),
    ("r2", "weak_contact",  "xgb", "stuff", "pitcher", 10),
    ("r2", "groundball",    "xgb", "stuff", "pitcher", 10),
    ("r2", "barrel",        "xgb", "stuff", "pitcher", 10),

    # --- Pitching+ / Out+ (combined) ---
    ("r2", "out",        "xgb", "pitching+", "pitcher", 10),
    ("r2", "xrun_value_re", "xgb", "pitching+", "pitcher", 10),
    ("r2", "xrun_value_re", "xgb", "pitching+", "season", 1),
    ("r2", "xrun_value_re", "xgb", "pitching+", "park", 1),
    ("r2", "xrun_value_re", "lgb", "pitching+", "pitcher", 10),

    # --- ablations / model comparison (whiff is the strongest signal) ---
    ("r2", "whiff", "xgb", "stuff+arsenal",  "pitcher", 10),
    ("r2", "whiff", "xgb", "stuff+location", "pitcher", 10),
    ("r2", "whiff", "lgb", "stuff",          "pitcher", 10),
]


def _run(job):
    tag, target, model, feats, holdout, folds = job
    return train.run_experiment(target, model, feats, holdout, folds,
                                n_jobs=2, tag=tag)


def main():
    print(f"launching {len(MATRIX)} R2 experiments on 4 workers ...")
    failures = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_run, j) for j in MATRIX]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                print(f"[{i}/{len(MATRIX)}] done", flush=True)
            except Exception as e:
                failures.append(e)
                print(f"[{i}/{len(MATRIX)}] FAILED: {e}", flush=True)
    if failures:
        print(f"*** {len(failures)} experiments FAILED — exiting non-zero ***", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
