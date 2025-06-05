"""Unified NRFI pipeline script."""

import argparse
import os
from datetime import datetime

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from pybaseball import statcast
import statsapi

DATA_DIR = "data_cache"
DATA_FILE = "final_training_data_clean_final.csv"
MODEL_FILE = "xgboost_yrfi_tuned.json"
PITCHER_SEASON_FILE = "pitcher_stats_season.csv"
TEAM_OFFENSE_FILE = "team_1st_inning_offense.csv"

# Manual pitcher ID mapping
manual_ids = {
    'brandon pfaadt': 680694,
    'grant holmes': 656550,
    'miles mikolas': 572070,
    'matthew liberatore': 669461,
    'chris bassitt': 605135,
    'framber valdez': 664285,
    'zach eflin': 621107,
    'dylan cease': 656546,
    'mitch keller': 641745,
    'jake irvin': 676600,
    'max fried': 621112,
    'jack leiter': 680678,
    'ryan pepiot': 675632
}

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


def train_model() -> None:
    """Tune and train the XGBoost model."""
    df = pd.read_csv(DATA_FILE).sort_values("season")
    X = df.drop(columns=["label"])
    y = df["label"]

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=500,
    )
    search = GridSearchCV(model, param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=1)
    search.fit(X, y)
    best_model = search.best_estimator_
    print("Best params:", search.best_params_)

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    best_model.fit(X_train, y_train)
    proba = best_model.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("AUC:", round(roc_auc_score(y_test, proba), 4))
    print("F1:", round(f1_score(y_test, preds), 4))
    print("LogLoss:", round(log_loss(y_test, proba), 4))

    best_model.get_booster().save_model(MODEL_FILE)
    print(f"Saved tuned model to {MODEL_FILE}")


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
    team_offense = pd.read_csv(TEAM_OFFENSE_FILE, parse_dates=["game_date"]).sort_values("game_date")
    latest_off = team_offense.groupby(["team", "half_inning"]).tail(1)
    team_offense_clean = latest_off.rename(
        columns={
            "OBP": "OBP_team",
            "SLG": "SLG_team",
            "K_rate": "K_rate_team",
            "BB_rate": "BB_rate_team",
            "runs_rolling10": "runs_rolling10_team",
        }
    )

    df["pitcher_key"] = df["pitcher_name"].str.strip().str.lower()
    df["pitcher"] = df["pitcher_key"].map(manual_ids)
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
    df = df.rename(columns={"team_x": "team", "team_y": "team_stats"})
    return df


def predict_today() -> pd.DataFrame:
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    expected_features = model.feature_names

    games = get_today_games()
    if games.empty:
        print("No games found for today")
        return pd.DataFrame()
    feats = prepare_features(games)
    X = pd.DataFrame({col: feats.get(col, 0) for col in expected_features})
    dmat = xgb.DMatrix(X[expected_features])
    feats["P_YRFI"] = model.predict(dmat)
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
    return feats[["team", "pitcher_name", "P_YRFI", "P_NRFI", "Confidence"]]


def main():
    parser = argparse.ArgumentParser(description="NRFI end-to-end pipeline")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("fetch_data", help="Download and cache Statcast data")
    sub.add_parser("train", help="Train the XGBoost model")
    sub.add_parser("predict_today", help="Predict YRFI/NRFI for today's games")
    args = parser.parse_args()

    if args.cmd == "fetch_data":
        current_year = datetime.now().year
        load_statcast_range(2023, current_year)
    elif args.cmd == "train":
        train_model()
    elif args.cmd == "predict_today":
        results = predict_today()
        if not results.empty:
            print(results.sort_values("P_YRFI", ascending=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
