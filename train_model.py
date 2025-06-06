import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from xgboost import XGBClassifier

DATA_FILE = "final_training_data_leakfree.csv"
PITCHER_ROLLING_FILE = "pitcher_rolling_stats.csv"
TEAM_OFFENSE_FILE = "team_1st_inning_offense.csv"
MODEL_FILE = "xgboost_yrfi_leakfree_tuned.json"
CALIBRATOR_FILE = "isotonic_calibrator.pkl"


def load_enhanced_dataset() -> pd.DataFrame:
    """Load base data and merge rolling statistics."""
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

    # days of rest since previous start for each pitcher
    df["days_rest"] = df.groupby("pitcher")["game_date"].diff().dt.days
    df["days_rest"].fillna(df["days_rest"].median(), inplace=True)

    off_cols = ["runs_rolling10_team", "OBP_team", "SLG_team", "K_rate_team", "BB_rate_team"]
    for c in off_cols:
        df[f"{c}_roll5"] = (
            df.groupby(["team", "half_inning"])[c].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )
        df[f"{c}_roll5"] = df[f"{c}_roll5"].fillna(
            df.groupby(["team", "half_inning"])[c].transform("median")
        )

    park_factors = {
        "AZ": 0.98,
        "ATL": 1.02,
        "BAL": 0.98,
        "BOS": 0.99,
        "CHC": 1.02,
        "CWS": 1.01,
        "CIN": 1.08,
        "CLE": 0.97,
        "COL": 1.33,
        "DET": 0.98,
        "HOU": 0.97,
        "KC": 1.01,
        "LAA": 1.04,
        "LAD": 1.00,
        "MIA": 0.95,
        "MIL": 1.02,
        "MIN": 1.02,
        "NYM": 1.02,
        "NYY": 1.03,
        "ATH": 0.96,
        "PHI": 1.05,
        "PIT": 0.99,
        "SD": 0.98,
        "SF": 0.97,
        "SEA": 0.97,
        "STL": 1.02,
        "TB": 0.97,
        "TEX": 1.03,
        "TOR": 1.05,
        "WSH": 1.02,
    }
    df["park_factor"] = df["team"].map(park_factors).fillna(1.0)

    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

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
        "OBP_team_roll5",
        "SLG_team_roll5",
        "K_rate_team_roll5",
        "BB_rate_team_roll5",
        "park_factor",
    ] + ["is_home_team", "label"]

    df = df[feature_cols]
    return df

def main():
    df = load_enhanced_dataset()
    df = df.sort_values('season')
    X = df.drop(columns=['label'])
    y = df['label']

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'n_estimators': [400, 700, 1000],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1.0, 1.5]
    }
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    search = GridSearchCV(model, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=1)
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
    print("Accuracy:", round(accuracy_score(y_test, preds),4))
    print("AUC:", round(roc_auc_score(y_test, proba),4))
    print("F1:", round(f1_score(y_test, preds),4))
    print("LogLoss:", round(log_loss(y_test, proba),4))

    best_model.get_booster().save_model(MODEL_FILE)
    pd.to_pickle(ir, CALIBRATOR_FILE)
    print(f"Saved tuned model to {MODEL_FILE} and calibrator to {CALIBRATOR_FILE}")

if __name__ == '__main__':
    main()
