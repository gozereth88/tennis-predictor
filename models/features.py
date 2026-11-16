"""Leak-free sequential feature builder. The same code path creates training rows and live
inference features, so there is no train/serve skew."""
import numpy as np
import pandas as pd
from utils.elo import EloEngine
from utils.preprocessing import SURFACES

CAT = ["surface", "level", "round", "hand_matchup"]
NUM = ["elo_diff", "surf_elo_diff", "form10_diff", "form30_diff", "activity_diff", "h2h_pct",
       "h2h_surf_pct", "h2h_n", "rank_diff", "log_rank_ratio", "rank_pts_diff", "age_diff",
       "height_diff", "best_of"]
FEATURES = CAT + NUM
DEFAULT_INFO = dict(rank=500.0, pts=0.0, age=25.0, hand="U", ht=185.0)


class FeatureState:
    def __init__(self):
        self.elo = EloEngine()
        self.hist, self.h2h, self.info, self.last_day = {}, {}, {}, 0

    def form(self, p, day):
        h = self.hist.get(p, [])
        last10 = h[-10:]
        rec = [w for d, w in h if day - d <= 30]
        return (sum(w for _, w in last10) + 2.5) / (len(last10) + 5), (sum(rec) + 1) / (len(rec) + 2), len(rec)

    def h2h_counts(self, a, b, surface=None):
        ss = SURFACES if surface is None else [surface]
        return (sum(self.h2h.get((a, b, s), 0) for s in ss), sum(self.h2h.get((b, a, s), 0) for s in ss))

    def latest(self, p, day):
        i = self.info.get(p)
        return dict(DEFAULT_INFO) if i is None else {**i, "age": i["age"] + (day - i["day"]) / 365.25}

    def features(self, p1, p2, ctx, day, i1, i2):
        s = ctx["surface"]
        f10a, f30a, na = self.form(p1, day)
        f10b, f30b, nb = self.form(p2, day)
        w, l = self.h2h_counts(p1, p2)
        sw, sl = self.h2h_counts(p1, p2, s)
        r = self.elo.rating
        return dict(
            surface=s, level=ctx["level"], round=ctx["round"], hand_matchup=f"{i1['hand']}v{i2['hand']}",
            elo_diff=r(p1, "all", day) - r(p2, "all", day), surf_elo_diff=r(p1, s, day) - r(p2, s, day),
            form10_diff=f10a - f10b, form30_diff=f30a - f30b, activity_diff=na - nb,
            h2h_pct=(w + 1) / (w + l + 2), h2h_surf_pct=(sw + 1) / (sw + sl + 2), h2h_n=w + l,
            rank_diff=i1["rank"] - i2["rank"], log_rank_ratio=float(np.log(i1["rank"] / i2["rank"])),
            rank_pts_diff=i1["pts"] - i2["pts"], age_diff=i1["age"] - i2["age"],
            height_diff=i1["ht"] - i2["ht"], best_of=int(ctx["best_of"]))

    def update(self, r):
        """Fold a finished match into state. Call only AFTER its features were extracted."""
        day = r.date.toordinal()
        w, l, s = r.winner_name, r.loser_name, r.surface
        self.elo.update(w, l, s, day, r.tourney_level)
        for p, won in ((w, 1), (l, 0)):
            h = self.hist.setdefault(p, [])
            h.append((day, won))
            del h[:-60]
        self.h2h[(w, l, s)] = self.h2h.get((w, l, s), 0) + 1
        for side, p in (("winner", w), ("loser", l)):
            self.info[p] = dict(rank=float(getattr(r, f"{side}_rank")), pts=float(getattr(r, f"{side}_rank_points")),
                                age=float(getattr(r, f"{side}_age")), hand=getattr(r, f"{side}_hand"),
                                ht=float(getattr(r, f"{side}_ht")), day=day)
        self.last_day = max(self.last_day, day)


def _info(r, side):
    return dict(rank=float(getattr(r, f"{side}_rank")), pts=float(getattr(r, f"{side}_rank_points")),
                age=float(getattr(r, f"{side}_age")), hand=getattr(r, f"{side}_hand"), ht=float(getattr(r, f"{side}_ht")))


def build_dataset(df: pd.DataFrame):
    """One chronological pass -> symmetric (augmented) rows: winner-as-P1 (y=1) and loser-as-P1 (y=0)."""
    state, rows = FeatureState(), []
    for mid, r in enumerate(df.itertuples(index=False)):
        day = r.date.toordinal()
        ctx = dict(surface=r.surface, level=r.tourney_level, round=r.round, best_of=r.best_of)
        iw, il = _info(r, "winner"), _info(r, "loser")
        a = state.features(r.winner_name, r.loser_name, ctx, day, iw, il)
        b = state.features(r.loser_name, r.winner_name, ctx, day, il, iw)
        a.update(y=1, mid=mid, date=r.date)
        b.update(y=0, mid=mid, date=r.date)
        rows += [a, b]
        state.update(r)
    return pd.DataFrame(rows), state
