"""Ingest + clean Jeff Sackmann-style ATP/WTA match CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd

ROUND_ORDER = {"Q1": 0, "Q2": 0, "Q3": 0, "R128": 1, "R64": 2, "R32": 3, "RR": 4, "R16": 4,
               "QF": 5, "BR": 6, "SF": 6, "F": 7}
SURFACES = ["Hard", "Clay", "Grass"]
_SIDE = ("name", "hand", "ht", "age", "rank", "rank_points")
REQUIRED = (["tourney_date", "tourney_name", "tourney_level", "surface", "round", "best_of",
             "match_num", "score"] + [f"{s}_{c}" for s in ("winner", "loser") for c in _SIDE])


def load_matches(tour: str, data_dir: str = "data") -> pd.DataFrame:
    files = sorted(Path(data_dir).glob(f"{tour}_matches_[0-9]*.csv"))
    if not files:
        raise FileNotFoundError(f"No {tour}_matches_YYYY.csv files in '{data_dir}/'. "
                                "Run `python data/download_data.py` first.")
    return clean_matches(pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True))


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED:
        if c not in df:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df["surface"] = df["surface"].replace({"Carpet": "Hard"})
    df = df[df["surface"].isin(SURFACES)].dropna(subset=["date", "winner_name", "loser_name"])
    df = df[~df["score"].fillna("").str.contains("W/O|DEF|Default", case=False)]  # drop walkovers
    for c in ("age", "ht"):
        med = pd.concat([df[f"winner_{c}"], df[f"loser_{c}"]]).median()
        med = {"age": 25.0, "ht": 185.0}[c] if pd.isna(med) else med
        for s in ("winner", "loser"):
            df[f"{s}_{c}"] = pd.to_numeric(df[f"{s}_{c}"], errors="coerce").fillna(med)
    for s in ("winner", "loser"):
        df[f"{s}_rank"] = pd.to_numeric(df[f"{s}_rank"], errors="coerce").fillna(1500)
        df[f"{s}_rank_points"] = pd.to_numeric(df[f"{s}_rank_points"], errors="coerce").fillna(0)
        df[f"{s}_hand"] = df[f"{s}_hand"].fillna("U").where(df[f"{s}_hand"].isin(["R", "L"]), "U")
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
    df["tourney_level"] = df["tourney_level"].fillna("A").astype(str)
    df["tourney_name"] = df["tourney_name"].fillna("")
    df["round"] = df["round"].fillna("R32").astype(str)
    df["round_ord"] = df["round"].map(ROUND_ORDER).fillna(3)
    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce").fillna(0)
    return df.sort_values(["date", "round_ord", "match_num"], kind="mergesort").reset_index(drop=True)
