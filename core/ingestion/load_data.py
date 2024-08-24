import urllib.request
from urllib.error import HTTPError

import certifi
import numpy as np
import pandas as pd
from tqdm import tqdm

import config.league as LEAGUE
import io
import os
import logging

from config.data_path import get_league_csv_paths
from core.preprocessing.league_preprocessing import feature_engineering_league
from core.preprocessing.preprocessing import calculate_h2h_stats
# from core.preprocessing.season import preprocessing_season
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

def encode_result(x):
    if str(x) == "H":
        return 1
    elif str(x) == "A":
        return 2
    elif str(x) == "D":
        return 0
    else:
        raise AttributeError(f'No match result value found for >> {x} << ')
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

    league_df = league_df.rename(columns={'FTHG': 'home_goals',
                                          'FTAG': 'away_goals',
                                          'FTR': 'result_1X2',
                                          'B365H': 'bet_1',
                                          'B365D': 'bet_X',
                                          'B365A': 'bet_2'})
    league_df['result_1X2'] = league_df['result_1X2'].apply(encode_result)
    league_df = league_df.reset_index(drop=True)
    return league_df
