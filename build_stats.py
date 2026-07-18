#!/usr/bin/env python3
"""
build_stats.py — nflverse -> one small team stats CSV
=====================================================
Runs inside GitHub Actions (free). Downloads nflverse play-by-play
for the current season and the prior season, computes the advanced
team stat table, and writes data/nfl_team_stats.csv.

Stats produced (offense + defense, per team, per season):
  - EPA/play (off + def)
  - Success rate (off + def), pass success %, rush success %
  - Explosive play % (rush 10+, pass 20+ yds) off + def
  - Red zone TD % (off + def)  [drives reaching opp 20]
  - Third down % (off + def)
  - Sack rate allowed (off) + sack rate (def)
  - Turnovers/game (off) + takeaways/game (def)
  - Last-5-games EPA + success rate (off + def)

Pressure rate is intentionally absent: it comes from charting data
that nflverse does not publish in play-by-play. Defensive sack rate
is the closest public proxy.
"""

import datetime
import os
import sys
import tempfile
import urllib.request

import pandas as pd

PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "pbp/play_by_play_{year}.parquet")

COLS = [
    "season", "week", "game_id", "posteam", "defteam", "epa", "success",
    "pass", "rush", "yards_gained", "qb_dropback", "sack",
    "third_down_converted", "third_down_failed",
    "interception", "fumble_lost", "yardline_100",
    "fixed_drive", "fixed_drive_result",
]


def season_year(today=None):
    """NFL season label: Aug-Dec -> current year, Jan-Jul -> prior year."""
    d = today or datetime.date.today()
    return d.year if d.month >= 8 else d.year - 1


def load_pbp(year):
    url = PBP_URL.format(year=year)
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp = f.name
        urllib.request.urlretrieve(url, tmp)
        df = pd.read_parquet(tmp, columns=COLS)
        os.unlink(tmp)
        return df
    except Exception as e:  # file not published yet (offseason) etc.
        print(f"  ! could not load {year}: {e}")
        return None


def team_table(df, season):
    """Compute the full stat table for one season of pbp."""
    # Real offensive snaps only
    plays = df[((df["pass"] == 1) | (df["rush"] == 1)) & df["epa"].notna()].copy()
    plays["explosive"] = (
        ((df["rush"] == 1) & (df["yards_gained"] >= 10)) |
        ((df["pass"] == 1) & (df["yards_gained"] >= 20))
    ).astype(int)

    teams = sorted(plays["posteam"].dropna().unique())
    rows = []

    # Red zone drives: any drive that had a snap inside the opp 20
    rz = plays[plays["yardline_100"] <= 20]
    rz_drives = rz.groupby(["game_id", "posteam", "fixed_drive"])["fixed_drive_result"] \
                  .first().reset_index()
    rz_drives["td"] = (rz_drives["fixed_drive_result"] == "Touchdown").astype(int)

    for tm in teams:
        off = plays[plays["posteam"] == tm]
        de = plays[plays["defteam"] == tm]
        games = off["game_id"].nunique()
        if games == 0:
            continue

        # last 5 games by week
        last5_ids = (off[["game_id", "week"]].drop_duplicates()
                     .sort_values("week").tail(5)["game_id"])
        off5 = off[off["game_id"].isin(last5_ids)]
        de5 = de[de["game_id"].isin(last5_ids)]

        def rate(num, den):
            return round(num / den, 4) if den else 0.0

        off_rz = rz_drives[rz_drives["posteam"] == tm]
        def_rz = rz_drives[rz_drives["posteam"] != tm]
        def_rz = def_rz.merge(
            de[["game_id"]].drop_duplicates(), on="game_id", how="inner")
        # defensive RZ = drives *against* this team in this team's games
        def_rz = rz_drives.merge(
            de[["game_id", "posteam"]].drop_duplicates()
              .rename(columns={"posteam": "opp"}),
            left_on=["game_id", "posteam"], right_on=["game_id", "opp"])

        rows.append({
            "season": season,
            "team": tm,
            "games": games,
            "epa_off": round(off["epa"].mean(), 4),
            "epa_def": round(de["epa"].mean(), 4),
            "sr_off": round(off["success"].mean(), 4),
            "sr_def": round(de["success"].mean(), 4),
            "pass_sr_off": round(off[off["pass"] == 1]["success"].mean(), 4),
            "rush_sr_off": round(off[off["rush"] == 1]["success"].mean(), 4),
            "expl_off": round(off["explosive"].mean(), 4),
            "expl_def": round(de["explosive"].mean(), 4),
            "rz_td_off": rate(off_rz["td"].sum(), len(off_rz)),
            "rz_td_def": rate(def_rz["td"].sum(), len(def_rz)),
            "third_off": rate(off["third_down_converted"].sum(),
                              off["third_down_converted"].sum()
                              + off["third_down_failed"].sum()),
            "third_def": rate(de["third_down_converted"].sum(),
                              de["third_down_converted"].sum()
                              + de["third_down_failed"].sum()),
            "sack_rate_allowed": rate(off["sack"].sum(), off["qb_dropback"].sum()),
            "sack_rate_def": rate(de["sack"].sum(), de["qb_dropback"].sum()),
            "to_pg": round((off["interception"].sum()
                            + off["fumble_lost"].sum()) / games, 3),
            "takeaways_pg": round((de["interception"].sum()
                                   + de["fumble_lost"].sum()) / games, 3),
            "l5_epa_off": round(off5["epa"].mean(), 4) if len(off5) else 0.0,
            "l5_epa_def": round(de5["epa"].mean(), 4) if len(de5) else 0.0,
            "l5_sr_off": round(off5["success"].mean(), 4) if len(off5) else 0.0,
            "l5_sr_def": round(de5["success"].mean(), 4) if len(de5) else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    cur = season_year()
    frames = []
    for yr in (cur - 1, cur):
        print(f"Loading {yr} play-by-play ...")
        df = load_pbp(yr)
        if df is None or df.empty:
            continue
        frames.append(team_table(df, yr))
        print(f"  {yr}: {len(frames[-1])} teams")

    if not frames:
        print("No data available for either season — aborting without changes.")
        sys.exit(0)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/nfl_team_stats.csv", index=False)
    print(f"Wrote data/nfl_team_stats.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
