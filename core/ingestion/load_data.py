import io
import urllib.request
from urllib.error import HTTPError

import certifi
import pandas as pd
from tqdm import tqdm

from config.data_path import get_league_csv_paths
from core.preprocessing.h2h_stats import calculate_h2h_stats
# from core.preprocessing.preprocessing import calculate_h2h_stats
from core.preprocessing.season import preprocessing_season_optimized
from core.time_decorator import timing


def extract_season_data(path, season_i, league_name, windows=None, next_matches=None):
    loading = False

    context = urllib.request.ssl.create_default_context(cafile=certifi.where())

    while not loading:
        try:
            with urllib.request.urlopen(path, context=context) as response:
                data = response.read().decode('utf-8')
                season_df = pd.read_csv(io.StringIO(data))
                season = int(path.split('/')[-2])
                # season_df = pd.read_csv(path, index_col=0)
                loading = True
        except HTTPError as err:
            print(f'Http error: {err}')

    season_df = season_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                             'FTR', 'B365H', 'B365D', 'B365A']]
    season_df['Date'] = pd.to_datetime(season_df['Date'], errors='coerce', dayfirst=True)
    if next_matches is not None:
        next_matches = next_matches.rename(columns={'date': 'Date'})
        next_matches['FTHG'] = None
        next_matches['FTAG'] = None
        next_matches['FTR'] = 'UNKNOWN'
        next_matches['B365H'] = None
        next_matches['B365D'] = None
        next_matches['B365A'] = None
        season_df = pd.concat((season_df, next_matches.drop('match_day', axis=1))).reset_index(drop=True)

    season_df = preprocessing_season_optimized(season_df, season, league_name, windows)

    season_df = season_df.reset_index(drop=True)

    return season_df


@timing
def extract_data(league_name, windows):
    league_df = pd.DataFrame()
    league_paths = get_league_csv_paths(league_name)
    for season_i, path in tqdm(enumerate(league_paths),
                               desc=' > Extracting Season Data: ',
                               total=len(league_paths)):
        season_df = extract_season_data(path, season_i, league_name, windows)
        league_df = pd.concat((league_df, season_df), axis=0)

    league_df = calculate_h2h_stats(league_df, {'league_name': league_name,
                                                'windows': windows})
    # for window in windows:
    #     league_df[[f'H2H_HomeWinRate_{window}', f'H2H_AwayWinRate_{window}',
    #                f'H2H_GoalDifference_{window}']] = league_df.apply(
    #         lambda row: calculate_h2h_stats(league_df, row['HomeTeam'], row['AwayTeam'], row['Date'], window), axis=1)

    league_df = league_df.reset_index(drop=True)
    return league_df
