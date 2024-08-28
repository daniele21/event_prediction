import io
import urllib.request
from urllib.error import HTTPError

import certifi
import pandas as pd
from tqdm import tqdm

from config.data_path import get_league_csv_paths
from core.preprocessing.preprocessing import calculate_h2h_stats
from core.preprocessing.season import preprocessing_season_optimized


def extract_season_data(path, season_i, league_name, windows=None):
    loading = False

    context = urllib.request.ssl.create_default_context(cafile=certifi.where())

    while not loading:
        try:
            with urllib.request.urlopen(path, context=context) as response:
                data = response.read().decode('utf-8')
                season_df = pd.read_csv(io.StringIO(data))
                season = path.split('/')[-2]
                # season_df = pd.read_csv(path, index_col=0)
                loading = True
        except HTTPError as err:
            print(f'Http error: {err}')

    # season_df = preprocessing_season(season_df, season_i, league_name)
    season_df = preprocessing_season_optimized(season_df, season, league_name, windows)

    # dropping nan rows
    # season_df = season_df.dropna()

    season_df = season_df.reset_index(drop=True)

    return season_df

def extract_data(league_name, windows):
    league_df = pd.DataFrame()
    league_paths = get_league_csv_paths(league_name)
    for season_i, path in tqdm(enumerate(league_paths),
                               desc=' > Extracting Season Data: ',
                               total=len(league_paths)):
        season_df = extract_season_data(path, season_i, league_name, windows)
        league_df = pd.concat((league_df, season_df), axis=0)

    for window in windows:
        league_df[[f'H2H_HomeWinRate_{window}', f'H2H_AwayWinRate_{window}',
                   f'H2H_GoalDifference_{window}']] = league_df.apply(
            lambda row: calculate_h2h_stats(league_df, row['HomeTeam'], row['AwayTeam'], row['Date'], window), axis=1)

    league_df = league_df.reset_index(drop=True)
    return league_df
