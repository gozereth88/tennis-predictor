"""Usage: python -m models.train --tour atp [--min-year 2000] [--iterations 600]"""
import argparse
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from models.features import CAT, FEATURES, build_dataset
from utils.preprocessing import load_matches

ARTIFACTS = Path(__file__).parent / "artifacts"


def make_model(iterations):
    return CatBoostClassifier(iterations=iterations, depth=6, learning_rate=0.05, l2_leaf_reg=5,
                              cat_features=CAT, random_seed=42, verbose=0)


def score(y, p):
    return dict(roc_auc=roc_auc_score(y, p), accuracy=accuracy_score(y, p > 0.5), log_loss=log_loss(y, p))


def train(tour, data_dir="data", min_year=2000, iterations=600, n_splits=5):
    df = load_matches(tour, data_dir)
    X, state = build_dataset(df)                       # Elo/H2H/form state is built from ALL history
    X = X[X["date"].dt.year >= min_year].reset_index(drop=True)
    mids = np.sort(X["mid"].unique())
    cut = int(len(mids) * 0.85)
    dev_mids, hold_mids = mids[:cut], mids[cut:]
    dev, hold = X[X["mid"].isin(dev_mids)], X[X["mid"].isin(hold_mids)]

    cv = []  # time-series CV on match order; both perspectives of a match stay in the same fold
    for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dev_mids)):
        m = make_model(iterations).fit(dev[dev["mid"].isin(dev_mids[tr])][FEATURES],
                                       dev[dev["mid"].isin(dev_mids[tr])]["y"])
        v = dev[dev["mid"].isin(dev_mids[va])]
        cv.append(dict(fold=k + 1, **score(v["y"], m.predict_proba(v[FEATURES])[:, 1])))
        print(cv[-1])

    m = make_model(iterations).fit(dev[FEATURES], dev["y"])
    p_hold = m.predict_proba(hold[FEATURES])[:, 1]
    holdout = score(hold["y"], p_hold)
    print("Holdout:", holdout)

    final = make_model(iterations).fit(X[FEATURES], X["y"])   # refit on everything for serving
    bundle = dict(
        model=final, features=FEATURES, state=state, cv=pd.DataFrame(cv), holdout=holdout,
        holdout_preds=hold[["y", "surface", "level", "date"]].assign(p=p_hold).reset_index(drop=True),
        importance=dict(zip(FEATURES, final.get_feature_importance())),
        levels=sorted(df["tourney_level"].unique()), n_matches=len(df), tour=tour,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        matches=df[["date", "winner_name", "loser_name", "surface", "tourney_name", "round", "score"]])
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(bundle, ARTIFACTS / f"{tour}_bundle.joblib", compress=3)
    print(f"Saved {ARTIFACTS / f'{tour}_bundle.joblib'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", choices=["atp", "wta"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-year", type=int, default=2000)
    ap.add_argument("--iterations", type=int, default=600)
    a = ap.parse_args()
    train(a.tour, a.data_dir, a.min_year, a.iterations)
