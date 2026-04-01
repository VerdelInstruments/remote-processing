#!/usr/bin/env python3
"""
RANSAC bridge: reads (tof, swim) arrays from JSON on stdin,
runs sklearn RANSACRegressor with production parameters,
writes {slope, intercept, inlier_count} to stdout.

Usage from Rust:
    echo '{"tof": [...], "swim": [...], "residual_threshold": 10, "random_state": 42}' | python3 ransac_bridge.py
"""
import json
import sys
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

req = json.load(sys.stdin)
tof = np.array(req["tof"])
swim = np.array(req["swim"])
threshold = req.get("residual_threshold", 10)
seed = req.get("random_state", 42)

if len(tof) < 2:
    json.dump({"slope": 0.0, "intercept": 0.0, "inlier_count": len(tof)}, sys.stdout)
    sys.exit(0)

model = RANSACRegressor(
    estimator=LinearRegression(),
    residual_threshold=threshold,
    max_trials=1000,
    min_samples=2,
    stop_n_inliers=req.get("stop_n_inliers", 50),
    random_state=seed,
)
model.fit(tof.reshape(-1, 1), swim)

json.dump({
    "slope": float(model.estimator_.coef_[0]),
    "intercept": float(model.estimator_.intercept_),
    "inlier_count": int(model.inlier_mask_.sum()),
}, sys.stdout)
