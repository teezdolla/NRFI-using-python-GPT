import pandas as pd
import statsapi
from pybaseball import playerid_lookup
import xgboost as xgb
from datetime import datetime
from functools import lru_cache
import argparse

# Load tuned booster
model = xgb.Booster()
model.load_model('xgboost_yrfi_tuned.json')
expected_features = model.feature_names

@lru_cache(maxsize=None)
def lookup_pitcher_id(name: str) -> int | None:
    """Resolve pitcher name to MLBAM id."""
    try:
        result = statsapi.lookup_player(name)
        if result:
            return int(result[0]["id"])
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

# Load stats CSVs
pitcher_season = pd.read_csv('pitcher_stats_season.csv')
team_offense = pd.read_csv('team_1st_inning_offense.csv', parse_dates=['game_date'])
pitcher_roll = pd.read_csv('pitcher_rolling_stats.csv', parse_dates=['game_date_start'])

# Latest offense stats per team/half-inning
team_offense = team_offense.sort_values(['team','half_inning','game_date'])
off_cols = ['runs_1st','OBP','SLG','K_rate','BB_rate']
for c in off_cols:
    team_offense[f'{c}_roll5'] = team_offense.groupby(['team','half_inning'])[c].transform(lambda s: s.rolling(5, min_periods=1).mean().shift())

latest_off = team_offense.groupby(['team', 'half_inning']).tail(1)
team_offense_clean = latest_off.rename(columns={
    'OBP':'OBP_team', 'SLG':'SLG_team',
    'K_rate':'K_rate_team', 'BB_rate':'BB_rate_team',
    'runs_rolling10':'runs_rolling10_team',
    'runs_1st_roll5':'runs_team_roll5', 'OBP_roll5':'OBP_team_roll5',
    'SLG_roll5':'SLG_team_roll5', 'K_rate_roll5':'K_rate_team_roll5',
    'BB_rate_roll5':'BB_rate_team_roll5'
})

pitcher_roll = pitcher_roll.sort_values(['pitcher','game_date_start'])
stats_cols = ['hits_allowed','walks','strikeouts','batters_faced','runs_allowed']
for c in stats_cols:
    pitcher_roll[f'{c}_roll3'] = pitcher_roll.groupby('pitcher')[c].transform(lambda s: s.rolling(3, min_periods=1).mean().shift())
pitcher_roll_latest = pitcher_roll.groupby('pitcher').tail(1)[['pitcher','game_date_start'] + [f'{c}_roll3' for c in stats_cols]]
pitcher_roll_latest['days_rest'] = (pd.Timestamp.today().normalize() - pitcher_roll_latest['game_date_start']).dt.days
pitcher_roll_latest = pitcher_roll_latest.drop(columns=['game_date_start'])

# Map team full names to abbreviations as in CSV
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

def get_today_games():
    today = datetime.today().strftime('%Y-%m-%d')
    schedule = statsapi.schedule(date=today)
    rows = []
    for game in schedule:
        home_pitcher = game.get('home_probable_pitcher')
        away_pitcher = game.get('away_probable_pitcher')
        if home_pitcher and away_pitcher:
            rows.append({
                'game_pk': game['game_id'],
                'game_date': today,
                'team': game['away_name'],
                'pitcher_name': home_pitcher,
                'inning_topbot': 'Top'
            })
            rows.append({
                'game_pk': game['game_id'],
                'game_date': today,
                'team': game['home_name'],
                'pitcher_name': away_pitcher,
                'inning_topbot': 'Bot'
            })
    df = pd.DataFrame(rows)
    df['half_inning'] = df['inning_topbot'] + '_1st'
    return df

def prepare_features(df):
    df['pitcher'] = df['pitcher_name'].apply(lookup_pitcher_id)
    df = df.dropna(subset=['pitcher']).copy()
    df['season'] = datetime.today().year
    df['is_home_team'] = df['inning_topbot'] == 'Bot'
    df['inning'] = 1
    # merge season stats
    season_cols = ['IDfg','Season','ERA','WHIP','FIP','K/9','BB/9','xFIP','CSW%','xERA']
    season_stats = pitcher_season[season_cols]
    season_stats = season_stats.rename(columns={
        'ERA':'ERA_season','WHIP':'WHIP_season','FIP':'FIP_season','K/9':'K/9_season',
        'BB/9':'BB/9_season','xFIP':'xFIP_season','CSW%':'CSW%_season','xERA':'xERA_season'
    })
    df = df.merge(season_stats, left_on=['pitcher','season'], right_on=['IDfg','Season'], how='left')
    df['team_abbr'] = df['team'].map(team_map)
    df = df.merge(team_offense_clean, left_on=['team_abbr','half_inning'], right_on=['team','half_inning'], how='left')
    df = df.merge(pitcher_roll_latest, on='pitcher', how='left')
    df = df.rename(columns={'team_x':'team','team_y':'team_stats'})
    return df

def predict(df):
    X = pd.DataFrame()
    for col in expected_features:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0
    X = X[expected_features]
    dmat = xgb.DMatrix(X)
    df['P_YRFI'] = model.predict(dmat)
    df['P_NRFI'] = 1 - df['P_YRFI']
    def label_conf(p):
        if p >= 0.75:
            return "🔥 High YRFI"
        elif p >= 0.65:
            return "⚠️ Moderate YRFI"
        elif p <= 0.35:
            return "🧊 Moderate NRFI"
        elif p <= 0.25:
            return "❄️ High NRFI"
        else:
            return "❓ Low Confidence"
    df['Confidence'] = df['P_YRFI'].apply(label_conf)
    return df[['game_pk','team','pitcher_name','inning_topbot','P_YRFI','P_NRFI','Confidence']]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict YRFI/NRFI for today's games")
    parser.add_argument('--output', help='CSV file to save results')
    parser.add_argument('--txt-output', help='Text file to save results')
    args = parser.parse_args()

    games = get_today_games()
    if games.empty:
        print('No games found for today')
    else:
        feats = prepare_features(games)
        results = predict(feats)

        # aggregate to full first inning probability
        pivot = results.pivot(index='game_pk', columns='inning_topbot', values='P_YRFI')
        pivot = pivot.rename(columns={'Top': 'P_YRFI_top', 'Bot': 'P_YRFI_bot'})
        pivot['P_YRFI'] = 1 - (1 - pivot.get('P_YRFI_top', 0)) * (1 - pivot.get('P_YRFI_bot', 0))
        pivot['P_NRFI'] = 1 - pivot['P_YRFI']

        info = {
            'away_team': games.loc[games['inning_topbot']=='Top'].set_index('game_pk')['team'],
            'home_team': games.loc[games['inning_topbot']=='Bot'].set_index('game_pk')['team'],
            'home_pitcher': games.loc[games['inning_topbot']=='Top'].set_index('game_pk')['pitcher_name'],
            'away_pitcher': games.loc[games['inning_topbot']=='Bot'].set_index('game_pk')['pitcher_name']
        }
        info_df = pd.concat(info, axis=1)
        full_results = pivot.merge(info_df, left_index=True, right_index=True)

        def label_conf(p):
            if p >= 0.75:
                return "🔥 High YRFI"
            elif p >= 0.65:
                return "⚠️ Moderate YRFI"
            elif p <= 0.35:
                return "🧊 Moderate NRFI"
            elif p <= 0.25:
                return "❄️ High NRFI"
            else:
                return "❓ Low Confidence"

        full_results['Confidence'] = full_results['P_YRFI'].apply(label_conf)
        full_results = full_results.reset_index()[['away_team','home_team','away_pitcher','home_pitcher','P_YRFI','P_NRFI','Confidence']]
        full_results = full_results.sort_values('P_YRFI', ascending=False)

        print(full_results)
        if args.output:
            full_results.to_csv(args.output, index=False)
        if args.txt_output:
            with open(args.txt_output, 'w') as f:
                f.write(full_results.to_string(index=False))
