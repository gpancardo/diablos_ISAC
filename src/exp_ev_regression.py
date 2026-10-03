"""Isolated test: regress ExitSpeed (continuous) instead of classifying
barrel/weak_contact (binary), per the "Exit Velocity Over Expected" approach
used in recent public pitch-quality work. Richer target -> maybe more signal.

Same pitcher-holdout GroupKFold discipline as train.py, same "stuff" feature
set, so it's an apples-to-apples comparison against results/phd__barrel__xgb__stuff__pitcher.json
(AUC 0.6148) and results/phd__weak_contact__xgb__stuff_clean__pitcher.json (AUC 0.6624).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

import features
import targets

df = targets.build_targets(targets.load())
batted = df[df["play_result"].isin(targets.BATTED_IN_PLAY_RESULTS)
           & df["ExitSpeed"].notna() & df["Angle"].notna()].reset_index(drop=True)
cat_levels = features.fit_categorical_levels(df)

groups = pd.factorize(batted["pitcher_anon_id"])[0]
gkf = GroupKFold(n_splits=10)

y = batted["ExitSpeed"].astype(float).values
y_barrel = ((batted["ExitSpeed"] >= 95) & (batted["Angle"] >= 20) & (batted["Angle"] <= 40)).astype(int).values
y_weak = (batted["ExitSpeed"] < 80).astype(int).values

oof_pred = np.full(len(batted), np.nan)
for tr, te in gkf.split(np.arange(len(batted)), groups=groups):
    baseline_df = df.iloc[np.where(np.isin(df["pitcher_anon_id"], batted.iloc[tr]["pitcher_anon_id"].unique()))[0]]
    X_train, names = features.build_X(batted.iloc[tr], baseline_df, "stuff", cat_levels)
    X_test, _ = features.build_X(batted.iloc[te], baseline_df, "stuff", cat_levels)
    m = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                     reg_lambda=1.0, tree_method="hist", n_jobs=3, random_state=0, verbosity=0)
    m.fit(X_train, y[tr])
    oof_pred[te] = m.predict(X_test)

rmse = float(np.sqrt(np.mean((oof_pred - y) ** 2)))
null_rmse = float(np.sqrt(np.mean((y.mean() - y) ** 2)))
spearman = float(pd.Series(oof_pred).corr(pd.Series(y), method="spearman"))
auc_barrel_from_ev = float(roc_auc_score(y_barrel, oof_pred))     # threshold the continuous pred
auc_weak_from_ev = float(roc_auc_score(y_weak, -oof_pred))        # lower pred EV = weaker contact

print(f"ExitSpeed regression: RMSE={rmse:.3f} (null={null_rmse:.3f}, "
      f"ratio={rmse/null_rmse:.4f}), Spearman={spearman:.4f}")
print(f"AUC (barrel, from continuous EV pred):       {auc_barrel_from_ev:.4f}  "
      f"vs direct classifier 0.6148")
print(f"AUC (weak_contact, from continuous EV pred): {auc_weak_from_ev:.4f}  "
      f"vs direct classifier 0.6624")
