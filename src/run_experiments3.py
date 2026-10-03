"""Round-3: ensemble (xgb+lgb) + arsenal ablation + fold-std reliability."""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

import train

# (tag, target, model, features, holdout, n_folds)
MATRIX = [
    # ensemble (variance reduction -> reliability/generalization)
    ("r3", "whiff",         "ensemble", "stuff",     "pitcher", 10),
    ("r3", "whiff",         "ensemble", "pitching+", "pitcher", 10),
    ("r3", "out",           "ensemble", "pitching+", "pitcher", 10),
    ("r3", "xrun_value_re", "ensemble", "pitching+", "pitcher", 10),
    ("r3", "xrun_value_re", "ensemble", "pitching+", "season", 1),
    ("r3", "xrun_value_re", "ensemble", "pitching+", "park", 1),
    # arsenal ablation (does arsenal help run value / out?)
    ("r3", "xrun_value_re", "xgb", "pitching+_noarsenal", "pitcher", 10),
    ("r3", "out",           "xgb", "pitching+_noarsenal", "pitcher", 10),
]


def _run(job):
    tag, target, model, feats, holdout, folds = job
    return train.run_experiment(target, model, feats, holdout, folds,
                                n_jobs=2, tag=tag)


def main():
    print(f"launching {len(MATRIX)} R3 experiments on 4 workers ...")
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
