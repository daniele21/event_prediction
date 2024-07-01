from urllib.error import HTTPError
import numpy as np
import pandas as pd
import config.league as LEAGUE


def preprocessing_season(season_df, n_season, league_name):
    data = season_df.copy(deep=True)

    data.insert(0, 'season', n_season)
    data.insert(0, 'league', league_name)
    data.insert(2, 'match_n', np.arange(1, len(data) + 1, 1))
    # data = _addRound(data)

    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    return data

def _extract_season_data(path, season_i, league_name):
    loading = False

    while(loading == False):
        try:
            season_df = pd.read_csv(path, index_col=0)
            loading = True
        except HTTPError as err:
            print(f'Http error: {err}')

    season_df = preprocessing_season(season_df, season_i, league_name)

    # dropping nan rows
    # season_df = season_df.dropna()

    season_df = season_df.reset_index(drop=True)

    return season_df


def extract_data(league_name):

    league_df = pd.DataFrame()

    for season_i, path in enumerate(LEAGUE.LEAGUE_PATHS[league_name]):
        season_df = _extract_season_data(path, season_i, league_name)
        league_df = league_df.append(season_df, sort=False)
        league_df = league_df.reset_index(drop=True)

    return league_df