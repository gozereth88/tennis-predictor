"""Download yearly match files from Jeff Sackmann's repos (CC BY-NC-SA 4.0, non-commercial use).
Usage: python data/download_data.py [--start 2000] [--end 2025]"""
import argparse, urllib.error, urllib.request
from datetime import date
from pathlib import Path

out = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=2000)
ap.add_argument("--end", type=int, default=date.today().year)
a = ap.parse_args()
for tour in ("atp", "wta"):
    for y in range(a.start, a.end + 1):
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{y}.csv"
        try:
            urllib.request.urlretrieve(url, out / f"{tour}_matches_{y}.csv")
            print("ok  ", tour, y)
        except urllib.error.HTTPError as e:
            print("skip", tour, y, e.code)
