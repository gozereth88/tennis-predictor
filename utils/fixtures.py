"""Daily ATP/WTA fixtures from ESPN's public (unofficial, keyless) scoreboard feed."""
import difflib, json, re, unicodedata, urllib.request
from datetime import datetime, timedelta, timezone
import pandas as pd

URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates={d}"
COLS = ["tour", "tournament", "round_raw", "round", "p1", "p2", "state", "winner", "start"]
ROUNDS = [("round of 128", "R128"), ("round of 64", "R64"), ("round of 32", "R32"), ("round of 16", "R16"),
          ("quarter", "QF"), ("semi", "SF"), ("round robin", "RR"), ("1st round", "R32"),
          ("2nd round", "R16"), ("3rd round", "R16"), ("4th round", "R16"), ("final", "F")]


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def _round(text):
    t = (text or "").lower()
    return next((code for key, code in ROUNDS if key in t), "R32")


def parse_scoreboard(data, tour, day):
    lo = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    hi = lo + timedelta(hours=30)  # covers late US evening starts that roll into the next UTC day
    rows = []
    for ev in data.get("events") or []:
        name = ev.get("name") or ev.get("shortName") or ""
        comps = [(c, "") for c in ev.get("competitions") or []]
        for g in ev.get("groupings") or []:
            gname = (g.get("grouping") or {}).get("displayName", "")
            comps += [(c, gname) for c in g.get("competitions") or []]
        for c, gname in comps:
            if "double" in gname.lower():
                continue
            cs = sorted(c.get("competitors") or [], key=lambda x: x.get("order", 0))
            if len(cs) != 2 or not all("athlete" in x for x in cs):
                continue  # doubles/teams or malformed
            names = [(x["athlete"].get("displayName") or "").strip() for x in cs]
            if any(not n or n.upper().startswith(("TBD", "BYE", "QUALIFIER")) for n in names):
                continue
            try:
                start = datetime.fromisoformat(c["date"].replace("Z", "+00:00"))
            except Exception:
                start = None
            if start and not (lo <= start < hi):
                continue
            state = ((c.get("status") or {}).get("type") or {}).get("state", "pre")
            win = next((i + 1 for i, x in enumerate(cs) if x.get("winner")), None)
            rnd = (c.get("round") or {}).get("displayName") or ""
            rows.append([tour, name, rnd, _round(rnd), names[0], names[1], state, win,
                         start.strftime("%H:%M") if start else ""])
    return pd.DataFrame(rows, columns=COLS)


def fetch_fixtures(tour, day):
    req = urllib.request.Request(URL.format(tour=tour, d=day.strftime("%Y%m%d")),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        return parse_scoreboard(json.load(r), tour, day)


class NameMatcher:
    """Map ESPN display names (accents, hyphens) onto the training-data player names."""
    def __init__(self, names):
        self.idx = {norm(n): n for n in names}
        self.keys = list(self.idx)

    def find(self, name):
        k = norm(name)
        if k in self.idx:
            return self.idx[k]
        m = difflib.get_close_matches(k, self.keys, n=1, cutoff=0.9)
        return self.idx[m[0]] if m else None


def surface_lookup(matches):
    recent = matches[matches["date"] >= matches["date"].max() - pd.Timedelta(days=3 * 365)]
    return {norm(k): v for k, v in recent.groupby("tourney_name")["surface"].last().items()}


def guess_surface(lookup, event_name):
    k = norm(event_name)
    if k in lookup:
        return lookup[k]
    m = difflib.get_close_matches(k, list(lookup), n=1, cutoff=0.7)
    if m:
        return lookup[m[0]]
    hit = next((v for n, v in lookup.items() if n and (n in k or k in n)), None)
    return hit or "Hard"


def guess_level(tour, event_name, levels):
    n = event_name.lower()
    if any(s in n for s in ("australian open", "roland garros", "french open", "wimbledon", "us open")):
        c = "G"
    elif "finals" in n:
        c = "F"
    elif "davis" in n or "billie jean" in n:
        c = "D"
    elif "masters" in n or "1000" in n:
        c = "M" if tour == "atp" else "PM"
    elif tour == "wta":
        c = "P" if "500" in n else "I"
    else:
        c = "A"
    return c if c in levels else ("A" if "A" in levels else levels[0])
