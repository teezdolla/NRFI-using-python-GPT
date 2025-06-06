"""Unified NRFI pipeline script."""

import argparse
import os
from datetime import datetime

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from pybaseball import statcast, playerid_lookup
import statsapi
from functools import lru_cache

DATA_DIR = "data_cache"
DATA_FILE = "final_training_data_leakfree.csv"
MODEL_FILE = "xgboost_yrfi_leakfree_tuned.json"
CALIBRATOR_FILE = "isotonic_calibrator.pkl"
PITCHER_SEASON_FILE = "pitcher_stats_season.csv"
TEAM_OFFENSE_FILE = "team_1st_inning_offense.csv"
PITCHER_ROLLING_FILE = "pitcher_rolling_stats.csv"


@lru_cache(maxsize=None)
def lookup_pitcher_id(name: str) -> int | None:
    try:
        res = statsapi.lookup_player(name)
        if res:
            return int(res[0]["id"])
    except Exception:
        pass
    try:
        last = name.split()[-1]
        first = name.split()[0]
        tbl = playerid_lookup(last, first)
        if not tbl.empty:
            return int(tbl["key_mlbam"].iloc[0])
    except Exception:
        pass
    return None


# Team name to abbreviation mapping
team_map = {
    'Arizona Diamondbacks': 'AZ',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'ATH',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSH'
}

park_factors = {
    'AZ': 0.98,
    'ATL': 1.02,
    'BAL': 0.98,
    'BOS': 0.99,
    'CHC': 1.02,
    'CWS': 1.01,
    'CIN': 1.08,
    'CLE': 0.97,
    'COL': 1.33,
    'DET': 0.98,
    'HOU': 0.97,
    'KC': 1.01,
    'LAA': 1.04,
    'LAD': 1.00,
    'MIA': 0.95,
    'MIL': 1.02,
    'MIN': 1.02,
    'NYM': 1.02,
    'NYY': 1.03,
    'ATH': 0.96,
    'PHI': 1.05,
    'PIT': 0.99,
    'SD': 0.98,
    'SF': 0.97,
    'SEA': 0.97,
    'STL': 1.02,
    'TB': 0.97,
    'TEX': 1.03,
    'TOR': 1.05,
    'WSH': 1.02,
}


def fetch_statcast_season(year: int) -> pd.DataFrame:
    """Download statcast data for a given season with caching."""
    os.makedirs(DATA_DIR, exist_ok=True)
    start = f"{year}-03-01"
    end = f"{year}-11-30"
    csv_path = os.path.join(DATA_DIR, f"statcast_{year}.csv")
    if os.path.exists(csv_path):
        print(f"Using cached data for {year}")
        return pd.read_csv(csv_path)
    print(f"Fetching statcast data for {year}...")
    df = statcast(start_dt=start, end_dt=end)
    df.to_csv(csv_path, index=False)
    return df


def load_statcast_range(start_year: int, end_year: int) -> pd.DataFrame:
    frames = [fetch_statcast_season(y) for y in range(start_year, end_year + 1)]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows")
    return df


def load_enhanced_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["game_date"]).sort_values("game_date")

    p_roll = pd.read_csv(PITCHER_ROLLING_FILE, parse_dates=["game_date_start"]).sort_values([
        "pitcher",
        "game_date_start",
    ])
    stats_cols = ["hits_allowed", "walks", "strikeouts", "batters_faced", "runs_allowed"]
    for c in stats_cols:
        p_roll[f"{c}_roll3"] = p_roll.groupby("pitcher")[c].transform(lambda s: s.rolling(3, min_periods=1).mean().shift())
    p_feats = p_roll[["pitcher", "game_date_start"] + [f"{c}_roll3" for c in stats_cols]]
    df = df.sort_values(["game_date", "pitcher"], kind="mergesort").reset_index(drop=True)
    p_feats = p_feats.sort_values(["game_date_start", "pitcher"], kind="mergesort").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        p_feats,
        left_on="game_date",
        right_on="game_date_start",
        by="pitcher",
        direction="backward",
    )
    df["days_rest"] = df.groupby("pitcher")["game_date"].diff().dt.days
    df["days_rest"].fillna(df["days_rest"].median(), inplace=True)

    t_off = pd.read_csv(TEAM_OFFENSE_FILE, parse_dates=["game_date"]).sort_values([
        "team",
        "half_inning",
        "game_date",
    ])
    off_cols = ["runs_rolling10", "OBP", "SLG", "K_rate", "BB_rate"]
    for c in off_cols:
        t_off[f"{c}_roll5"] = t_off.groupby(["team", "half_inning"])[c].transform(lambda s: s.rolling(5, min_periods=1).mean().shift())
    t_feats = t_off[["team", "half_inning", "game_date"] + [f"{c}_roll5" for c in off_cols]]
    df = df.sort_values(["game_date", "team", "half_inning"], kind="mergesort").reset_index(drop=True)
    t_feats = t_feats.sort_values(["game_date", "team", "half_inning"], kind="mergesort").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        t_feats,
        left_on="game_date",
        right_on="game_date",
        by=["team", "half_inning"],
        direction="backward",
    )

    medians = df.median(numeric_only=True)
    df = df.fillna(medians)
    rename_map = {f"{c}_roll5": f"{c}_team_roll5" for c in off_cols}
    df = df.rename(columns=rename_map)
    df["park_factor"] = df["team"].map(park_factors).fillna(1.0)

    feature_cols = [
        "inning",
        "pitcher",
        "season",
        "days_rest",
    ] + [f"{c}_roll3" for c in stats_cols] + [
        "ERA_season",
        "WHIP_season",
        "FIP_season",
        "K/9_season",
        "BB/9_season",
        "xFIP_season",
        "CSW%_season",
        "xERA_season",
        "runs_rolling10_team",
        "OBP_team",
        "SLG_team",
        "K_rate_team",
        "BB_rate_team",
    ] + list(rename_map.values()) + ["park_factor", "is_home_team", "label"]

    return df[feature_cols]


def train_model() -> None:
    """Tune and train the XGBoost model."""
    df = load_enhanced_dataset().sort_values("season")
    X = df.drop(columns=["label"])
    y = df["label"]

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "n_estimators": [400, 700, 1000],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1.0, 1.5],
    }
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    search = GridSearchCV(model, param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=1)
    search.fit(X, y)
    best_model = search.best_estimator_
    print("Best params:", search.best_params_)

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False,
    )
    proba_raw = best_model.predict_proba(X_test)[:, 1]
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(proba_raw, y_test)
    proba = ir.predict(proba_raw)
    preds = (proba > 0.5).astype(int)
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("AUC:", round(roc_auc_score(y_test, proba), 4))
    print("F1:", round(f1_score(y_test, preds), 4))
    print("LogLoss:", round(log_loss(y_test, proba), 4))

    best_model.get_booster().save_model(MODEL_FILE)
    pd.to_pickle(ir, CALIBRATOR_FILE)
    print(f"Saved tuned model to {MODEL_FILE} and calibrator to {CALIBRATOR_FILE}")


def get_today_games() -> pd.DataFrame:
    """Retrieve today's scheduled games with probable pitchers."""
    today = datetime.today().strftime("%Y-%m-%d")
    schedule = statsapi.schedule(date=today)
    rows = []
    for game in schedule:
        home_pitcher = game.get("home_probable_pitcher")
        away_pitcher = game.get("away_probable_pitcher")
        if home_pitcher and away_pitcher:
            rows.append({
                "game_pk": game["game_id"],
                "game_date": today,
                "team": game["away_name"],
                "pitcher_name": home_pitcher,
                "inning_topbot": "Top",
            })
            rows.append({
                "game_pk": game["game_id"],
                "game_date": today,
                "team": game["home_name"],
                "pitcher_name": away_pitcher,
                "inning_topbot": "Bot",
            })
    df = pd.DataFrame(rows)
    df["half_inning"] = df["inning_topbot"] + "_1st"
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge cached statistics to build the model feature set."""
    pitcher_season = pd.read_csv(PITCHER_SEASON_FILE)
    team_offense = pd.read_csv(TEAM_OFFENSE_FILE, parse_dates=["game_date"]).sort_values([
        "team",
        "half_inning",
        "game_date",
    ])
    off_cols = ["runs_rolling10", "OBP", "SLG", "K_rate", "BB_rate"]
    for c in off_cols:
        team_offense[f"{c}_roll5"] = team_offense.groupby(["team", "half_inning"])[c].transform(
            lambda s: s.rolling(5, min_periods=1).mean().shift()
        )
    latest_off = team_offense.groupby(["team", "half_inning"]).tail(1)
    team_offense_clean = latest_off.rename(
        columns={
            "OBP": "OBP_team",
            "SLG": "SLG_team",
            "K_rate": "K_rate_team",
            "BB_rate": "BB_rate_team",
            "runs_rolling10": "runs_rolling10_team",
            "OBP_roll5": "OBP_team_roll5",
            "SLG_roll5": "SLG_team_roll5",
            "K_rate_roll5": "K_rate_team_roll5",
            "BB_rate_roll5": "BB_rate_team_roll5",
        }
    )

    pitcher_roll = pd.read_csv(PITCHER_ROLLING_FILE, parse_dates=["game_date_start"]).sort_values([
        "pitcher",
        "game_date_start",
    ])
    stats_cols = ["hits_allowed", "walks", "strikeouts", "batters_faced", "runs_allowed"]
    for c in stats_cols:
        pitcher_roll[f"{c}_roll3"] = pitcher_roll.groupby("pitcher")[c].transform(
            lambda s: s.rolling(3, min_periods=1).mean().shift()
        )
    pitcher_roll_latest = pitcher_roll.groupby("pitcher").tail(1)[["pitcher"] + [f"{c}_roll3" for c in stats_cols]]

    df["pitcher"] = df["pitcher_name"].apply(lookup_pitcher_id)
    df = df.dropna(subset=["pitcher"]).copy()
    df["season"] = datetime.today().year
    df["is_home_team"] = df["inning_topbot"] == "Bot"
    df["inning"] = 1

    season_cols = ["IDfg", "Season", "ERA", "WHIP", "FIP", "K/9", "BB/9", "xFIP", "CSW%", "xERA"]
    season_stats = pitcher_season[season_cols].rename(
        columns={
            "ERA": "ERA_season",
            "WHIP": "WHIP_season",
            "FIP": "FIP_season",
            "K/9": "K/9_season",
            "BB/9": "BB/9_season",
            "xFIP": "xFIP_season",
            "CSW%": "CSW%_season",
            "xERA": "xERA_season",
        }
    )
    df = df.merge(season_stats, left_on=["pitcher", "season"], right_on=["IDfg", "Season"], how="left")
    df["team_abbr"] = df["team"].map(team_map)
    df = df.merge(team_offense_clean, left_on=["team_abbr", "half_inning"], right_on=["team", "half_inning"], how="left")
    df = df.merge(pitcher_roll_latest, on="pitcher", how="left")
    df = df.rename(columns={"team_x": "team", "team_y": "team_stats"})
    df["park_factor"] = df["team_abbr"].map(park_factors).fillna(1.0)
    return df


def predict_today(output_csv: str | None = None, output_txt: str | None = None) -> pd.DataFrame:
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    calibrator = pd.read_pickle(CALIBRATOR_FILE)
    expected_features = model.feature_names

    games = get_today_games()
    if games.empty:
        print("No games found for today")
        return pd.DataFrame()
    feats = prepare_features(games)
    X = pd.DataFrame({col: feats.get(col, 0) for col in expected_features})
    dmat = xgb.DMatrix(X[expected_features])
    raw_proba = model.predict(dmat)
    feats["P_YRFI"] = calibrator.predict(raw_proba)
    feats["P_NRFI"] = 1 - feats["P_YRFI"]

    def label_conf(p: float) -> str:
        if p >= 0.75:
            return "🔥 High YRFI"
        if p >= 0.65:
            return "⚠️ Moderate YRFI"
        if p <= 0.35:
            return "🧊 Moderate NRFI"
        if p <= 0.25:
            return "❄️ High NRFI"
        return "❓ Low Confidence"

    feats["Confidence"] = feats["P_YRFI"].apply(label_conf)
    results = feats[["team", "pitcher_name", "P_YRFI", "P_NRFI", "Confidence"]].sort_values("P_YRFI", ascending=False)
    if output_csv:
        results.to_csv(output_csv, index=False)
    if output_txt:
        with open(output_txt, "w") as f:
            f.write(results.to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description="NRFI end-to-end pipeline")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("fetch_data", help="Download and cache Statcast data")
    sub.add_parser("train", help="Train the XGBoost model")
    pred = sub.add_parser("predict_today", help="Predict YRFI/NRFI for today's games")
    pred.add_argument('--output', help='CSV file to save results')
    pred.add_argument('--txt-output', help='Text file to save results')
    args = parser.parse_args()

    if args.cmd == "fetch_data":
        current_year = datetime.now().year
        load_statcast_range(2023, current_year)
    elif args.cmd == "train":
        train_model()
    elif args.cmd == "predict_today":
        results = predict_today(args.output, args.txt_output)
        if not results.empty:
            print(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
