# Tennis Match Predictor (ATP + WTA)

Streamlit app backed by CatBoost, sequential surface-aware Elo, form, H2H and ranking features.

## Layout
```
app.py                 Streamlit UI (4 tabs)
data/download_data.py  Fetch Sackmann CSVs -> data/{atp,wta}_matches_YYYY.csv
utils/elo.py           Time-decayed overall + surface Elo
utils/preprocessing.py Ingestion, cleaning, chronological ordering
models/features.py     Leak-free sequential feature builder (+ symmetric augmentation)
models/train.py        Time-series CV, holdout eval, joblib bundle export
models/predict.py      Inference helpers (symmetrised probabilities)
models/artifacts/      <tour>_bundle.joblib (model, Elo/H2H state, metrics, match log)
```

## Local setup
```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10-3.12
pip install -r requirements.txt
python data/download_data.py --start 2000
python -m models.train --tour atp
python -m models.train --tour wta
streamlit run app.py
```
Data: Jeff Sackmann's `tennis_atp` / `tennis_wta` (CC BY-NC-SA 4.0, **non-commercial** use, attribution required).

## How leakage is prevented
Matches are processed once in date/round order. Features are read from the running state *before* the
match updates Elo, form, H2H and player info. Each match yields two rows (winner-as-P1 y=1, loser-as-P1 y=0),
and CV splits by match so both rows always land in the same fold. At inference, P(P1) averages both
perspectives, removing any positional bias.

## Updating predictions with new results
Re-run `download_data.py` and `models.train`; the bundle stores the latest Elo/form/H2H state, and Elo decay
is evaluated relative to the last match date in your data.

## Streamlit Community Cloud
1. Train locally, then commit `models/artifacts/*.joblib` (compressed; keep each under GitHub's 100 MB limit,
   or use Git LFS). Raw CSVs are not needed at runtime.
2. Push the repo to GitHub, go to share.streamlit.io, choose the repo, set the main file to `app.py`.
3. Use Python 3.11 in Advanced settings; dependencies install from `requirements.txt`.

## Notes
Players are keyed by name. Fixtures with unknown names are still predicted from default priors and flagged.
Tournament level codes follow the data (ATP: G, M, A, F, D; WTA adds PM, P, I). Betting odds are not used;
expect roughly 64-68% accuracy, in line with strong public models.
