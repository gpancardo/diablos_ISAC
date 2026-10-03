"""Wave 2: model families (MLP/GAM), spin-efficiency feature, model comparison."""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

import train

# (tag, target, model, features, holdout, n_folds)
MATRIX = [
    # model families on whiff (strongest stuff signal)
    ("w2", "whiff", "mlp", "stuff",          "pitcher", 10),
    ("w2", "whiff", "gam", "stuff",          "pitcher", 5),
    # spin-efficiency feature (H4)
    ("w2", "whiff",         "xgb", "stuff+spin",    "pitcher", 10),
    ("w2", "whiff",         "xgb", "pitching+spin", "pitcher", 10),
    ("w2", "xrun_value_re", "xgb", "pitching+spin", "pitcher", 10),
    # model families on run value
    ("w2", "xrun_value_re", "mlp", "pitching+", "pitcher", 10),
    # model families on out
    ("w2", "out", "mlp", "pitching+", "pitcher", 10),
    ("w2", "out", "lgb", "pitching+", "pitcher", 10),
]


def _run(job):
    tag, target, model, feats, holdout, folds = job
    return train.run_experiment(target, model, feats, holdout, folds,
                                n_jobs=2, tag=tag)


def main():
    print(f"launching {len(MATRIX)} Wave-2 experiments on 4 workers ...")
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
