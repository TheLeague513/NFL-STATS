#!/usr/bin/env python3
"""
tune_weights.py — fit the model's Settings weights from real games
==================================================================
Instead of eyeballed dial values, this fits them by regression:

  1. Downloads nflverse play-by-play for 2023-2025 and the nflverse
     games file (scores + closing lines).
  2. For every regular-season game in weeks 5-18, computes each
     team's SEASON-TO-DATE stats entering that game — the same
     differentials the Google Sheet uses live (no peeking at the
     game being predicted).
  3. Regresses actual home margin on those differentials. The
     coefficients ARE the Settings weights; the intercept is HFA.
  4. Validates honestly with leave-one-season-out cross-validation
     and compares the model's RMSE to the closing line's RMSE.
  5. Writes a paste-ready block to data/tuned_weights.txt.

Notes on reading the output:
  - Some coefficients may come out small or even negative. That is
    collinearity doing its job — EPA, success rate, and yards/play
    overlap heavily, so once one carries the signal the regression
    shrinks the redundant ones. Paste the values as-is; they
    jointly minimize error.
  - Expect the market RMSE to beat the model RMSE. Fitted weights
    close the gap; they don't flip it.
"""

import os
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd

SEASONS = [2023, 2024, 2025]
PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "pbp/play_by_play_{year}.parquet")
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
MIN_PRIOR_GAMES = 3

PBP_COLS = [
    "season", "week", "game_id", "posteam", "defteam", "epa", "success",
    "pass", "rush", "yards_gained", "qb_dropback", "sack",
    "third_down_converted", "third_down_failed",
    "interception", "fumble_lost", "yardline_100",
    "fixed_drive", "fixed_drive_result",
]

# feature key -> Settings row name in the Google Sheet
FEATURES = [
    ("ppg",  "Weight: Net PPG"),
    ("epa",  "Weight: EPA/Play"),
    ("sr",   "Weight: Success Rate"),
    ("expl", "Weight: Explosive %"),
    ("rz",   "Weight: Red Zone TD %"),
    ("sack", "Weight: Sack Rates"),
    ("ypp",  "Weight: Yards/Play"),
    ("t3",   "Weight: 3rd Down"),
    ("to",   "Weight: Turnover Margin"),
]


def load_pbp(year):
    url = PBP_URL.format(year=year)
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp = f.name
        urllib.request.urlretrieve(url, tmp)
        df = pd.read_parquet(tmp, columns=PBP_COLS)
        os.unlink(tmp)
        return df
    except Exception as e:
        print(f"  ! could not load {year} pbp: {e}")
        return None


def per_side(plays, side):
    """Aggregate one game's plays for the offense ('off') or defense."""
    key = "posteam" if side == "off" else "defteam"
    g = (plays.groupby(["season", "week", "game_id", key])
         .agg(epa=("epa", "sum"), n=("epa", "size"),
              succ=("success", "sum"), expl=("explosive", "sum"),
              yards=("yards_gained", "sum"),
              sacks=("sack", "sum"), db=("qb_dropback", "sum"),
              t3c=("third_down_converted", "sum"),
              t3f=("third_down_failed", "sum"),
              to=("turnover", "sum"))
         .reset_index().rename(columns={key: "team"}))
    suf = "_o" if side == "off" else "_d"
    return g.rename(columns={c: c + suf for c in
                             ["epa", "n", "succ", "expl", "yards",
                              "sacks", "db", "t3c", "t3f", "to"]})


def red_zone(plays, games):
    """Per-game red-zone trips + TDs for offense and defense."""
    rz = plays[plays["yardline_100"] <= 20]
    drives = (rz.groupby(["season", "week", "game_id", "posteam", "fixed_drive"])
              ["fixed_drive_result"].first().reset_index())
    drives["td"] = (drives["fixed_drive_result"] == "Touchdown").astype(int)
    off = (drives.groupby(["season", "week", "game_id", "posteam"])
           .agg(rzt_o=("td", "size"), rztd_o=("td", "sum")).reset_index()
           .rename(columns={"posteam": "team"}))
    # defense = the same drives, credited to the opponent
    opp = games[["game_id", "home_team", "away_team"]]
    d = off.merge(opp, on="game_id", how="left")
    d["team2"] = np.where(d["team"] == d["home_team"], d["away_team"], d["home_team"])
    de = d[["season", "week", "game_id", "team2", "rzt_o", "rztd_o"]].rename(
        columns={"team2": "team", "rzt_o": "rzt_d", "rztd_o": "rztd_d"})
    return off, de


def build_team_games(games):
    """Long table: one row per team per game with points for/against."""
    h = games.rename(columns={"home_team": "team", "home_score": "pf",
                              "away_score": "pa"})[
        ["season", "week", "game_id", "team", "pf", "pa"]]
    a = games.rename(columns={"away_team": "team", "away_score": "pf",
                              "home_score": "pa"})[
        ["season", "week", "game_id", "team", "pf", "pa"]]
    return pd.concat([h, a], ignore_index=True)


def main():
    print("Downloading games file ...")
    games = pd.read_csv(GAMES_URL)
    games = games[(games["season"].isin(SEASONS)) &
                  (games["game_type"] == "REG") &
                  games["home_score"].notna()].copy()
    games["result"] = games["home_score"] - games["away_score"]

    frames = []
    for yr in SEASONS:
        print(f"Loading {yr} play-by-play ...")
        pbp = load_pbp(yr)
        if pbp is None:
            continue
        plays = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1))
                    & pbp["epa"].notna()].copy()
        plays["explosive"] = (
            ((plays["rush"] == 1) & (plays["yards_gained"] >= 10)) |
            ((plays["pass"] == 1) & (plays["yards_gained"] >= 20))
        ).astype(int)
        plays["turnover"] = ((plays["interception"] == 1) |
                             (plays["fumble_lost"] == 1)).astype(int)

        yr_games = games[games["season"] == yr]
        off = per_side(plays, "off")
        de = per_side(plays, "def")
        rz_o, rz_d = red_zone(plays, yr_games)
        pts = build_team_games(yr_games)

        tg = (pts.merge(off, on=["season", "week", "game_id", "team"], how="left")
                 .merge(de, on=["season", "week", "game_id", "team"], how="left")
                 .merge(rz_o, on=["season", "week", "game_id", "team"], how="left")
                 .merge(rz_d, on=["season", "week", "game_id", "team"], how="left"))
        frames.append(tg)
        print(f"  {yr}: {len(tg)} team-games")

    if not frames:
        sys.exit("No play-by-play available — aborting.")

    tg = pd.concat(frames, ignore_index=True).fillna(0)
    tg = tg.sort_values(["season", "team", "week"]).reset_index(drop=True)

    # ---- season-to-date sums ENTERING each game (cumsum minus self) ----
    grp = tg.groupby(["season", "team"])
    sum_cols = [c for c in tg.columns
                if c.endswith("_o") or c.endswith("_d") or c in ("pf", "pa")]
    for c in sum_cols:
        tg["p_" + c] = grp[c].cumsum() - tg[c]
    tg["gp"] = grp.cumcount()

    def safe(numer, denom):
        return np.where(denom > 0, numer / np.maximum(denom, 1), 0.0)

    tg["f_ppg"] = safe(tg["p_pf"] - tg["p_pa"], tg["gp"])
    tg["f_epa"] = safe(tg["p_epa_o"], tg["p_n_o"]) - safe(tg["p_epa_d"], tg["p_n_d"])
    tg["f_sr"] = safe(tg["p_succ_o"], tg["p_n_o"]) - safe(tg["p_succ_d"], tg["p_n_d"])
    tg["f_expl"] = safe(tg["p_expl_o"], tg["p_n_o"]) - safe(tg["p_expl_d"], tg["p_n_d"])
    tg["f_rz"] = safe(tg["p_rztd_o"], tg["p_rzt_o"]) - safe(tg["p_rztd_d"], tg["p_rzt_d"])
    tg["f_sack"] = safe(tg["p_sacks_d"], tg["p_db_d"]) - safe(tg["p_sacks_o"], tg["p_db_o"])
    tg["f_ypp"] = safe(tg["p_yards_o"], tg["p_n_o"]) - safe(tg["p_yards_d"], tg["p_n_d"])
    tg["f_t3"] = (safe(tg["p_t3c_o"], tg["p_t3c_o"] + tg["p_t3f_o"])
                  - safe(tg["p_t3c_d"], tg["p_t3c_d"] + tg["p_t3f_d"]))
    tg["f_to"] = safe(tg["p_to_d"] - tg["p_to_o"], tg["gp"])

    feat_cols = ["f_" + k for k, _ in FEATURES]
    lookup = tg.set_index(["season", "week", "game_id", "team"])

    # ---- assemble the regression rows: one per game, weeks 5-18 ----
    rows = []
    for _, g in games[(games["week"] >= 5) & (games["week"] <= 18)].iterrows():
        try:
            h = lookup.loc[(g["season"], g["week"], g["game_id"], g["home_team"])]
            a = lookup.loc[(g["season"], g["week"], g["game_id"], g["away_team"])]
        except KeyError:
            continue
        if h["gp"] < MIN_PRIOR_GAMES or a["gp"] < MIN_PRIOR_GAMES:
            continue
        rows.append(
            [g["season"], g["result"], g["spread_line"]]
            + [h[c] - a[c] for c in feat_cols])
    df = pd.DataFrame(rows, columns=["season", "result", "close"] + feat_cols)
    print(f"\nRegression sample: {len(df)} games across {sorted(df['season'].unique())}")

    def fit(train):
        X = np.column_stack([np.ones(len(train))] + [train[c] for c in feat_cols])
        beta, _, _, _ = np.linalg.lstsq(X, train["result"].values, rcond=None)
        return beta

    def predict(beta, d):
        X = np.column_stack([np.ones(len(d))] + [d[c] for c in feat_cols])
        return X @ beta

    def rmse(err):
        return float(np.sqrt(np.mean(np.square(err))))

    # ---- leave-one-season-out cross-validation ----
    print("\nLeave-one-season-out validation (honest, out-of-sample):")
    cv_model, cv_market = [], []
    for s in sorted(df["season"].unique()):
        tr, te = df[df["season"] != s], df[df["season"] == s]
        if len(tr) == 0 or len(te) == 0:
            continue
        b = fit(tr)
        m = rmse(te["result"].values - predict(b, te))
        mk = rmse(te["result"].values - te["close"].values)
        cv_model.append(m)
        cv_market.append(mk)
        print(f"  test {s}: model RMSE {m:.2f}  |  closing line RMSE {mk:.2f}")
    print(f"  AVG    : model RMSE {np.mean(cv_model):.2f}  |  "
          f"closing line RMSE {np.mean(cv_market):.2f}")

    # ---- final fit on everything -> the weights you paste in ----
    beta = fit(df)
    lines = []
    lines.append("TUNED WEIGHTS — paste into the Settings tab")
    lines.append("=" * 46)
    lines.append(f"{'HFA (points)':30s} {beta[0]:8.2f}")
    for i, (key, name) in enumerate(FEATURES):
        lines.append(f"{name:30s} {beta[i + 1]:8.2f}")
    lines.append("")
    lines.append(f"Sample: {len(df)} games, weeks 5-18, seasons {SEASONS}")
    lines.append(f"Out-of-sample model RMSE : {np.mean(cv_model):.2f}")
    lines.append(f"Closing line RMSE        : {np.mean(cv_market):.2f}")
    lines.append("")
    lines.append("Notes: negative/small values are fine (collinearity —")
    lines.append("redundant stats get shrunk once EPA carries the signal).")
    lines.append("Paste all values as-is, including HFA.")

    out = "\n".join(lines)
    print("\n" + out)
    os.makedirs("data", exist_ok=True)
    with open("data/tuned_weights.txt", "w") as f:
        f.write(out + "\n")
    print("\nWrote data/tuned_weights.txt")


if __name__ == "__main__":
    main()
