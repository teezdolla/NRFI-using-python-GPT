import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from xgboost import XGBClassifier

DATA_FILE = "final_training_data.csv"
PITCHER_ROLLING_FILE = "pitcher_rolling_stats.csv"
TEAM_OFFENSE_FILE = "team_1st_inning_offense.csv"
MODEL_FILE = "xgboost_yrfi_tuned.json"


def load_enhanced_dataset() -> pd.DataFrame:
    """Load base data and merge rolling statistics."""
    df = pd.read_csv(DATA_FILE, parse_dates=["game_date"]).sort_values(["pitcher", "game_date"])

    # days of rest since previous start for each pitcher
    df["days_rest"] = df.groupby("pitcher")["game_date"].diff().dt.days
    df["days_rest"].fillna(df["days_rest"].median(), inplace=True)

    stats_cols = ["hits_allowed", "walks", "strikeouts", "batters_faced", "runs_allowed"]
    for c in stats_cols:
        df[f"{c}_roll3"] = (
            df.groupby("pitcher")[c].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
        )

    off_cols = ["runs_1st", "OBP_team", "SLG_team", "K_rate_team", "BB_rate_team"]
    for c in off_cols:
        df[f"{c}_roll5"] = (
            df.groupby(["team", "half_inning"])[c].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

    df = df.fillna(0)

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
        "runs_1st_roll5",
        "OBP_team_roll5",
        "SLG_team_roll5",
        "K_rate_team_roll5",
        "BB_rate_team_roll5",
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
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          random_state=42, n_estimators=500)
    search = GridSearchCV(model, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=1)
    search.fit(X, y)
    best_model = search.best_estimator_
    print("Best params:", search.best_params_)

    split = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    best_model.fit(X_train, y_train)
    proba = best_model.predict_proba(X_test)[:,1]
    preds = (proba > 0.5).astype(int)
    print("Accuracy:", round(accuracy_score(y_test, preds),4))
    print("AUC:", round(roc_auc_score(y_test, proba),4))
    print("F1:", round(f1_score(y_test, preds),4))
    print("LogLoss:", round(log_loss(y_test, proba),4))

    best_model.get_booster().save_model(MODEL_FILE)
    print(f"Saved tuned model to {MODEL_FILE}")

if __name__ == '__main__':
    main()
