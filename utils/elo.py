"""Time-decayed overall + surface-specific Elo, updated strictly match-by-match."""
INIT = 1500.0
LEVEL_MULT = {"G": 1.1, "M": 1.0, "PM": 1.0, "F": 1.0, "A": 0.95, "P": 0.95, "D": 0.9}


class EloEngine:
    def __init__(self, decay_start=90, half_life=365.0):
        self.decay_start, self.half_life = decay_start, half_life
        self.r, self.n, self.last = {}, {}, {}  # keyed by (player, "all" | surface)

    def rating(self, p, key, day):
        """Rating as of `day` (ordinal), regressed toward 1500 after long inactivity."""
        r = self.r.get((p, key), INIT)
        last = self.last.get((p, key))
        if last is not None:
            idle = max(0, day - last - self.decay_start)
            r = INIT + (r - INIT) * 0.5 ** (idle / self.half_life)
        return r

    def _k(self, p, key, level):  # FiveThirtyEight-style decaying K-factor
        return 250.0 / (self.n.get((p, key), 0) + 5) ** 0.4 * LEVEL_MULT.get(level, 0.95)

    def update(self, winner, loser, surface, day, level="A"):
        for key in ("all", surface):
            rw, rl = self.rating(winner, key, day), self.rating(loser, key, day)
            exp_w = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
            kw, kl = self._k(winner, key, level), self._k(loser, key, level)
            for p, new in ((winner, rw + kw * (1 - exp_w)), (loser, rl - kl * (1 - exp_w))):
                self.r[(p, key)] = new
                self.n[(p, key)] = self.n.get((p, key), 0) + 1
                self.last[(p, key)] = day
