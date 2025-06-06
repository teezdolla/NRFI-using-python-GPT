import os
from datetime import datetime
import pandas as pd
from pybaseball import statcast

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_statcast_season(year):
    """Download statcast data for a season with caching."""
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

def load_statcast_range(start_year, end_year):
    frames = [fetch_statcast_season(y) for y in range(start_year, end_year + 1)]
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    current_year = datetime.now().year
    df = load_statcast_range(2023, current_year)
    print(df.head())
