from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def model_tables(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, r2_score
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        import sys

        print(
            "[models] scikit-learn not installed; skipping predictive models. "
            "Install: pip install scikit-learn",
            file=sys.stderr,
        )
        return pd.DataFrame([])

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return pd.DataFrame([])

    X = df[emo_cols].fillna(0.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=7)

    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        y = df[metric]
        if y.isna().all():
            continue
        if metric == "stereotype_label":
            yb = y.dropna().astype(int)
            Xb = X.loc[yb.index]
            if yb.nunique() < 2 or len(yb) < 10:
                continue
            model = Pipeline(steps=[("scale", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=200))])
            aucs = []
            accs = []
            for train_idx, test_idx in kf.split(Xb):
                X_train, X_test = Xb.iloc[train_idx], Xb.iloc[test_idx]
                y_train, y_test = yb.iloc[train_idx], yb.iloc[test_idx]
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                aucs.append(roc_auc_score(y_test, prob))
                accs.append(accuracy_score(y_test, pred))
            rows.append(
                {
                    "metric": metric,
                    "model": "logistic_regression",
                    "n": int(len(yb)),
                    "roc_auc_mean": float(np.mean(aucs)),
                    "acc_mean": float(np.mean(accs)),
                }
            )
        else:
            yv = y.dropna().astype(float)
            Xv = X.loc[yv.index]
            if len(yv) < 10:
                continue
            model = Pipeline(steps=[("scale", StandardScaler(with_mean=False)), ("reg", Ridge(alpha=1.0))])
            r2s = []
            maes = []
            for train_idx, test_idx in kf.split(Xv):
                X_train, X_test = Xv.iloc[train_idx], Xv.iloc[test_idx]
                y_train, y_test = yv.iloc[train_idx], yv.iloc[test_idx]
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2s.append(r2_score(y_test, pred))
                maes.append(mean_absolute_error(y_test, pred))
            rows.append(
                {
                    "metric": metric,
                    "model": "ridge_regression",
                    "n": int(len(yv)),
                    "r2_mean": float(np.mean(r2s)),
                    "mae_mean": float(np.mean(maes)),
                }
            )

    return pd.DataFrame(rows)
