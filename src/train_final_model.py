"""Train and serialize the final whiff model + dashboard data.

Produces everything the Streamlit simulator needs:
  dashboard/models/whiff_model.joblib   trained xgb (stuff features)
  dashboard/models/feature_names.json   ordered feature columns
  dashboard/models/cat_levels.json      one-hot vocab
  dashboard/data/whiff_norm.csv         mean/std of P(whiff) per pitch type (for the
                                        100-scale), computed from OOF predictions
  dashboard/data/altitude_delta.csv     empirical Whiff+ delta per pitch type
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

import features
import targets

os.makedirs("dashboard/models", exist_ok=True)
os.makedirs("dashboard/data", exist_ok=True)

df = targets.build_targets(targets.load())
cat_levels = features.fit_categorical_levels(df)

# --- train final whiff model on ALL swings (stuff features) ---
swings = df[df["swing"] == 1]
X, names = features.build_X(swings, swings, "stuff", cat_levels)
y = swings["y_whiff"].values.astype(float)

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      tree_method="hist", n_jobs=4, random_state=0, verbosity=0,
                      eval_metric="logloss")
model.fit(X, y)
joblib.dump(model, "dashboard/models/whiff_model.joblib")
json.dump(names, open("dashboard/models/feature_names.json", "w"))
json.dump(cat_levels, open("dashboard/models/cat_levels.json", "w"))

# --- normalization stats from OOF predictions (100-scale) ---
oof = pd.read_parquet("results/r2__whiff__xgb__stuff__pitcher.oof.parquet")
norm = oof.groupby("pitch_type", observed=True)["pred"].agg(["mean", "std"]).reset_index()
norm.columns = ["pitch_type", "mean", "std"]
norm.to_csv("dashboard/data/whiff_norm.csv", index=False)

# --- altitude delta (empirical Whiff+ by pitch type) ---
alt = pd.read_csv("results/whiff_plus_by_arsenal.csv")
# pivot to delta table
piv = alt.pivot_table(index="pitch_type", columns="altitude", values="stuff_plus")
delta = pd.DataFrame({
    "pitch_type": piv.index,
    "whiff_plus_sea": piv["No Altitude"].values,
    "whiff_plus_extreme": piv["Extreme Altitude"].values,
})
delta["delta"] = delta["whiff_plus_extreme"] - delta["whiff_plus_sea"]
delta.to_csv("dashboard/data/altitude_delta.csv", index=False)

print("saved:", sorted(os.listdir("dashboard/models")) + sorted(os.listdir("dashboard/data")))
print("\n--- altitude delta (Whiff+) ---")
print(delta.round(2).to_string(index=False))
