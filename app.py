from datetime import date
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, streamlit as st
from sklearn.metrics import roc_curve
from models.predict import predict_match, profile
from utils.fixtures import NameMatcher, fetch_fixtures, guess_level, guess_surface, surface_lookup
from utils.preprocessing import SURFACES

st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="wide")
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR"]
LEVELS = {"G": "Grand Slam", "M": "Masters 1000", "PM": "Premier Mandatory", "P": "Premier", "A": "Tour 250/500",
          "I": "International", "F": "Tour Finals", "D": "Davis/BJK Cup"}


@st.cache_resource(show_spinner="Loading model…")
def load_bundle(tour):
    p = Path(__file__).parent / "models" / "artifacts" / f"{tour}_bundle.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def player_names(tour):
    b = load_bundle(tour); st_ = b["state"]
    recent = [p for p, i in st_.info.items() if st_.last_day - i["day"] < 3 * 365]
    return sorted(recent or st_.info)


@st.cache_data(ttl=900, show_spinner="Fetching fixtures…")
def get_fixtures(tour, day):
    return fetch_fixtures(tour, day)


@st.cache_resource
def tour_tools(tour):
    b = load_bundle(tour)
    return NameMatcher(b["state"].info.keys()), surface_lookup(b["matches"])


def level_label(c): return f"{LEVELS.get(c, c)} ({c})"


st.title("🎾 Tennis Match Predictor")
tour = st.sidebar.radio("Tour", ["atp", "wta"], format_func=str.upper)
bundle = load_bundle(tour)
if bundle is None:
    st.error(f"No trained {tour.upper()} model found. Run `python data/download_data.py` then "
             f"`python -m models.train --tour {tour}` and reload.")
    st.stop()
players, state = player_names(tour), bundle["state"]
st.sidebar.caption(f"Trained on {bundle['n_matches']:,} matches · data through "
                   f"{pd.Timestamp.fromordinal(state.last_day):%d %b %Y} · built {bundle['trained_at']}")

t1, t2, t3, t4 = st.tabs(["Match predictor", "Head-to-head", "Model performance", "Daily predictions"])

with t1:
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Player 1", players, index=0)
    p2 = c2.selectbox("Player 2", players, index=min(1, len(players) - 1))
    c3, c4, c5, c6 = st.columns(4)
    surface = c3.selectbox("Surface", SURFACES)
    level = c4.selectbox("Tournament level", bundle["levels"], format_func=level_label)
    rnd = c5.selectbox("Round", ROUNDS, index=4)
    bo = c6.radio("Best of", [3, 5], horizontal=True, index=1 if (level == "G" and tour == "atp") else 0)
    if st.button("Predict", type="primary"):
        if p1 == p2:
            st.error("Pick two different players.")
        else:
            try:
                prob, f = predict_match(bundle, p1, p2, surface, level, rnd, bo)
                a, b = st.columns(2)
                for col, name, pr in ((a, p1, prob), (b, p2, 1 - prob)):
                    col.metric(name, f"{pr:.1%}")
                    col.progress(float(pr))
                gap = abs(prob - 0.5)
                st.info(f"Pick: **{p1 if prob >= .5 else p2}** — "
                        f"{'high' if gap > .25 else 'moderate' if gap > .1 else 'low'} confidence")
                pa, pb = profile(bundle, p1, surface), profile(bundle, p2, surface)
                st.dataframe(pd.DataFrame({p1: pa, p2: pb}).astype(str), use_container_width=True)
                w, l = state.h2h_counts(p1, p2); sw, sl = state.h2h_counts(p1, p2, surface)
                st.write(f"**Head-to-head:** {p1} {w}–{l} {p2} overall · {sw}–{sl} on {surface.lower()}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with t2:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Player A", players, index=0, key="ha")
    b = c2.selectbox("Player B", players, index=min(1, len(players) - 1), key="hb")
    m = bundle["matches"]
    h = m[((m.winner_name == a) & (m.loser_name == b)) | ((m.winner_name == b) & (m.loser_name == a))]
    if a == b:
        st.warning("Choose two different players.")
    elif h.empty:
        st.info("These players have no recorded meetings in the dataset.")
    else:
        wa = (h.winner_name == a).sum()
        x, y = st.columns(2); x.metric(a, f"{wa} wins"); y.metric(b, f"{len(h) - wa} wins")
        by = h.assign(winner=h.winner_name).groupby(["surface", "winner"]).size().unstack(fill_value=0)
        st.bar_chart(by.reindex(columns=[a, b], fill_value=0))
        st.dataframe(h.sort_values("date", ascending=False).assign(date=lambda d: d.date.dt.date)
                     [["date", "tourney_name", "surface", "round", "winner_name", "score"]],
                     use_container_width=True, hide_index=True)

with t3:
    hm = bundle["holdout"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (holdout)", f"{hm['roc_auc']:.3f}")
    c2.metric("Accuracy", f"{hm['accuracy']:.1%}")
    c3.metric("Log loss", f"{hm['log_loss']:.3f}")
    st.caption("Holdout = most recent 15% of matches. Cross-validation (expanding window) per fold:")
    st.dataframe(bundle["cv"].round(4), hide_index=True, use_container_width=True)
    imp = pd.Series(bundle["importance"]).sort_values()
    st.subheader("Feature importance"); st.bar_chart(imp, horizontal=True)
    hp = bundle["holdout_preds"]
    g1, g2 = st.columns(2)
    fpr, tpr, _ = roc_curve(hp.y, hp.p)
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); g1.pyplot(fig)
    bins = pd.cut(hp.p, np.linspace(0, 1, 11)); cal = hp.groupby(bins, observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot(cal.pred, cal.obs, "o-"); ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set(xlabel="Predicted probability", ylabel="Observed win rate", title="Calibration"); g2.pyplot(fig)
    st.subheader("Accuracy by surface")
    st.dataframe(hp.assign(ok=(hp.p > .5) == (hp.y == 1)).groupby("surface").ok.mean().rename("accuracy").to_frame().style.format("{:.1%}"))

with t4:
    st.subheader("Daily predictions — ATP & WTA")
    st.caption("Singles fixtures come from ESPN's public scoreboard and cover both tours, whatever tour is picked in the sidebar.")
    day = st.date_input("Match day", value=date.today())
    bundles = {t: load_bundle(t) for t in ("atp", "wta")}
    frames = []
    for t in ("atp", "wta"):
        if bundles[t] is None:
            st.warning(f"No {t.upper()} model found, so {t.upper()} matches are skipped.")
            continue
        try:
            frames.append(get_fixtures(t, day))
        except Exception as e:
            st.warning(f"Could not load {t.upper()} fixtures ({e}). Try again in a minute.")
    fx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if fx.empty:
        st.info("No ATP or WTA tour-level singles matches found for this date. Tour events pause between "
                "seasons, so try a date during a tournament week.")
    else:
        all_levels = sorted(set().union(*[b["levels"] for b in bundles.values() if b]))
        tdf = fx[["tour", "tournament"]].drop_duplicates().reset_index(drop=True)
        tdf["surface"] = [guess_surface(tour_tools(r.tour)[1], r.tournament) for r in tdf.itertuples()]
        tdf["level"] = [guess_level(r.tour, r.tournament, bundles[r.tour]["levels"]) for r in tdf.itertuples()]
        st.caption("Surface and level are guessed from the tournament name. Correct them here if needed.")
        tdf = st.data_editor(tdf, hide_index=True, use_container_width=True, disabled=["tour", "tournament"],
                             key=f"ctx_{day}", column_config={
                                 "surface": st.column_config.SelectboxColumn(options=SURFACES, required=True),
                                 "level": st.column_config.SelectboxColumn(options=all_levels, required=True)})
        ctx = {(r.tour, r.tournament): (r.surface, r.level) for r in tdf.itertuples()}
        status = {"pre": "Upcoming", "in": "Live", "post": "Finished"}
        rows = []
        for r in fx.itertuples():
            bd, (sf, lv) = bundles[r.tour], ctx[(r.tour, r.tournament)]
            matcher = tour_tools(r.tour)[0]
            a, b = matcher.find(r.p1), matcher.find(r.p2)
            row = {"Tour": r.tour.upper(), "Tournament": r.tournament, "Round": r.round, "Time (UTC)": r.start,
                   "Status": status.get(r.state, r.state), "Player 1": r.p1, "Player 2": r.p2}
            if a and b and a != b:
                try:
                    pr, _ = predict_match(bd, a, b, sf, lv, r.round, 5 if (lv == "G" and r.tour == "atp") else 3)
                    row.update({"P1 win %": round(pr * 100, 1), "P2 win %": round((1 - pr) * 100, 1),
                                "Pick": r.p1 if pr >= .5 else r.p2})
                    if r.winner:
                        row["Result"] = "✔" if (pr >= .5) == (r.winner == 1) else "✘"
                except Exception as e:
                    row["Note"] = f"prediction failed: {e}"
            else:
                row["Note"] = "player not found in training data: " + ", ".join(
                    n for n, m in ((r.p1, a), (r.p2, b)) if not m)
            rows.append(row)
        res = pd.DataFrame(rows)
        shown = st.multiselect("Show", sorted(res["Status"].unique()), default=sorted(res["Status"].unique()))
        res = res[res["Status"].isin(shown)]
        done = res["Result"].dropna() if "Result" in res else []
        c1, c2 = st.columns(2)
        c1.metric("Matches", len(res))
        if len(done):
            c2.metric("Correct on finished matches", f"{(done == '✔').mean():.0%} ({len(done)})")
        st.dataframe(res, use_container_width=True, hide_index=True, column_config={
            "P1 win %": st.column_config.ProgressColumn("P1 win %", min_value=0, max_value=100, format="%.1f%%"),
            "P2 win %": st.column_config.ProgressColumn("P2 win %", min_value=0, max_value=100, format="%.1f%%")})
        st.download_button("Download predictions", res.to_csv(index=False), f"predictions_{day}.csv")
    with st.expander("Custom fixtures (CSV upload or manual entry)"):
        st.write("Upload a CSV or edit the table. Columns: `player1, player2, surface, level, round, best_of`.")
        tmpl = pd.DataFrame({"player1": [players[0]], "player2": [players[-1]], "surface": ["Hard"],
                             "level": [bundle["levels"][0]], "round": ["R32"], "best_of": [3]})
        st.download_button("Download template", tmpl.to_csv(index=False), "fixtures_template.csv")
        up = st.file_uploader("Fixtures CSV", type="csv")
        fx = pd.read_csv(up) if up else st.data_editor(tmpl, num_rows="dynamic", use_container_width=True)
        if st.button("Generate predictions", type="primary"):
            missing = {"player1", "player2", "surface", "level", "round", "best_of"} - set(fx.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
            else:
                out = []
                for r in fx.itertuples(index=False):
                    try:
                        if r.surface not in SURFACES:
                            raise ValueError(f"surface must be one of {SURFACES}")
                        pr, _ = predict_match(bundle, r.player1, r.player2, r.surface, r.level, r.round, int(r.best_of))
                        unk = [p for p in (r.player1, r.player2) if p not in state.info]
                        out.append(dict(player1=r.player1, player2=r.player2, p1_win=round(pr, 3), p2_win=round(1 - pr, 3),
                                        pick=r.player1 if pr >= .5 else r.player2,
                                        note=f"unknown player: {', '.join(unk)}" if unk else ""))
                    except Exception as e:
                        out.append(dict(player1=r.player1, player2=r.player2, note=f"error: {e}"))
                res = pd.DataFrame(out)
                st.dataframe(res, use_container_width=True, hide_index=True)
                st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv")
