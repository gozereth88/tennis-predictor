import pandas as pd
from models.features import FEATURES


def default_day(state):
    return state.last_day + 1  # day after latest training match (avoids spurious Elo decay on stale data)


def predict_match(bundle, p1, p2, surface, level, rnd, best_of, day=None):
    """Symmetrised P(p1 wins): average of the P1-view and the inverted P2-view."""
    st = bundle["state"]
    day = day or default_day(st)
    ctx = dict(surface=surface, level=level, round=rnd, best_of=best_of)
    f12 = st.features(p1, p2, ctx, day, st.latest(p1, day), st.latest(p2, day))
    f21 = st.features(p2, p1, ctx, day, st.latest(p2, day), st.latest(p1, day))
    p = bundle["model"].predict_proba(pd.DataFrame([f12, f21])[FEATURES])[:, 1]
    return float(0.5 * (p[0] + 1 - p[1])), f12


def profile(bundle, p, surface):
    st = bundle["state"]; day = default_day(st); i = st.latest(p, day); f10, f30, n30 = st.form(p, day)
    return {"Overall Elo": round(st.elo.rating(p, "all", day)), f"{surface} Elo": round(st.elo.rating(p, surface, day)),
            "Form (last 10)": f"{f10:.0%}", "Form (30 days)": f"{f30:.0%} ({n30} matches)",
            "Rank": int(i["rank"]), "Rank points": int(i["pts"]), "Age": round(i["age"], 1),
            "Height (cm)": int(i["ht"]), "Hand": i["hand"]}
