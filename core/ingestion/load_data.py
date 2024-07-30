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


from core.preprocessing.league_preprocessing import feature_engineering_league
from core.preprocessing.season import preprocessing_season


def extract_season_data(path, season_i, league_name):
    loading = False

    context = urllib.request.ssl.create_default_context(cafile=certifi.where())

    while not loading:
        try:
            with urllib.request.urlopen(path, context=context) as response:
                data = response.read().decode('utf-8')
                season_df = pd.read_csv(io.StringIO(data))

                # season_df = pd.read_csv(path, index_col=0)
                loading = True
        except HTTPError as err:
            print(f'Http error: {err}')

    season_df = preprocessing_season(season_df, season_i, league_name)

    # dropping nan rows
    # season_df = season_df.dropna()

    season_df = season_df.reset_index(drop=True)

    return season_df


def extract_data(league_name, n_prev_match):
    league_df = pd.DataFrame()

    for season_i, path in tqdm(enumerate(LEAGUE.LEAGUE_PATHS[league_name]),
                               desc=' > Extracting Season Data: '):
        season_df = extract_season_data(path, season_i, league_name)
        league_df = pd.concat((league_df, season_df), axis=0)
        league_df = league_df.reset_index(drop=True)

    league_df = feature_engineering_league(league_df, n_prev_match)

    return league_df
