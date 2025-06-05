import pandas as pd
import statsapi
import xgboost as xgb
from datetime import datetime

# Load tuned booster
model = xgb.Booster()
model.load_model('xgboost_yrfi_tuned.json')
expected_features = model.feature_names

# Manual pitcher ID mapping (from notebook)
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

# Load stats CSVs
pitcher_season = pd.read_csv('pitcher_stats_season.csv')
team_offense = pd.read_csv('team_1st_inning_offense.csv', parse_dates=['game_date'])

# Latest offense stats per team/half-inning
team_offense = team_offense.sort_values('game_date')
latest_off = team_offense.groupby(['team', 'half_inning']).tail(1)
team_offense_clean = latest_off.rename(columns={
    'OBP':'OBP_team', 'SLG':'SLG_team',
    'K_rate':'K_rate_team', 'BB_rate':'BB_rate_team',
    'runs_rolling10':'runs_rolling10_team'
})

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
    df['pitcher_key'] = df['pitcher_name'].str.strip().str.lower()
    df['pitcher'] = df['pitcher_key'].map(manual_ids)
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
    return df[['team','pitcher_name','P_YRFI','P_NRFI','Confidence']]

if __name__ == '__main__':
    games = get_today_games()
    if games.empty:
        print('No games found for today')
    else:
        feats = prepare_features(games)
        results = predict(feats)
        print(results.sort_values('P_YRFI', ascending=False))
