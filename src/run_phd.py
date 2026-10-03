"""PhD-quality experiment round (Sept 2026) — honest, reproducible, loud-failure.

Fixes the methodological gaps found in the audit and produces the numbers the
docs were missing:

  E1  whiff on `stuff_clean` (de-correlated) + ensemble  -> does AUC survive
      removing the 4 near-duplicate columns? (expect ~unchanged; SHAP becomes
      interpretable)
  E2  contact targets (weak_contact / groundball / barrel) with the CORRECTED
      batted-ball denominator (play_result in balls-in-play, not `is_batted`)
  E3  MLP + GAM on whiff  -> the "4 model families" claim, actually run

Unlike run_experiments*.py, this FAILS LOUDLY (exit 1) if any experiment errors,
so partial results can never be mistaken for a complete run.

Usage:
  .venv/bin/python src/run_phd.py e1
  .venv/bin/python src/run_phd.py e2
  .venv/bin/python src/run_phd.py e3
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import train

E1 = [
    # (target, model, features, holdout, n_folds)
    ("whiff", "xgb",      "stuff_clean", "pitcher", 10),
    ("whiff", "ensemble", "stuff_clean", "pitcher", 10),
]

E2 = [
    ("weak_contact", "xgb", "stuff",       "pitcher", 10),
    ("groundball",   "xgb", "stuff",       "pitcher", 10),
    ("barrel",       "xgb", "stuff",       "pitcher", 10),
    ("weak_contact", "xgb", "stuff_clean", "pitcher", 10),
    ("groundball",   "xgb", "stuff_clean", "pitcher", 10),
    ("barrel",       "xgb", "stuff_clean", "pitcher", 10),
]

E3 = [
    ("whiff", "mlp", "stuff",       "pitcher", 10),
    ("whiff", "gam", "stuff",       "pitcher", 5),
    ("whiff", "mlp", "stuff_clean", "pitcher", 10),
]

JOBS = {"e1": E1, "e2": E2, "e3": E3}


def run(jobs, tag):
    failures = []
    for i, (target, model, feats, holdout, folds) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {tag} {target} {model} {feats} {holdout} ===",
              flush=True)
        try:
            res = train.run_experiment(target, model, feats, holdout, folds,
                                       n_jobs=4, tag=tag)
            print(f"    mean = {res['mean']}", flush=True)
        except Exception as e:  # loud: record + abort with non-zero
            failures.append((target, model, feats, str(e)))
            import traceback
            traceback.print_exc()
    if failures:
        print("\n*** FAILURES (aborting, exit 1):", flush=True)
        for f in failures:
            print("   ", f, flush=True)
        sys.exit(1)
    print(f"\n{tag}: {len(jobs)} experiments OK", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e1"
    tag = sys.argv[2] if len(sys.argv) > 2 else "phd"
    if which not in JOBS:
        sys.exit(f"unknown group {which!r}; use e1|e2|e3")
    run(JOBS[which], tag)
