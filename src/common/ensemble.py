"""The deployed movie/TV regressor: an equal-weight mean of XGBoost, CatBoost and SVR.

One definition shared by the trainers (what ships) and the benchmarks (what is
reported), so the two cannot drift apart again. On the 50 registry folds it beat the
Optuna-tuned edge-penalty stack with ordinal member, crowd anchor and inner-CV it
replaced (movie skill 0.458 vs 0.429, TV 0.255 vs 0.229) at ~1/50th of the cost.
"""
import os

import catboost as ctb
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def model_threads():
    """Threads one model may use: all cores when serial, the worker's share inside map_folds."""
    return int(os.getenv("OMP_NUM_THREADS", 0)) or -1


def mean_ensemble():
    return VotingRegressor([
        ("xgb", xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42)),
        ("cat", ctb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=5, loss_function="MAE",
                                      verbose=0, random_seed=42, thread_count=model_threads())),
        ("svr", Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))])),
    ])
