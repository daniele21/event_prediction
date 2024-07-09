from os.path import exists

import pandas as pd

from pathlib import Path
import os
import config.league as LEAGUE

from config.data_path import get_league_csv_paths
from core.ingestion.load_data import extract_data, extract_season_data
from core.logger import logger
from core.preprocessing.data_shift import shift_data_features
from core.preprocessing.league_preprocessing import feature_engineering_league
from core.time_decorator import timing


class DatabaseManager:

    def __init__(self, params):
        self.params = params

    @timing
    def extract_data_league(self):
        league_name = self.params['league_name']
        n_prev_match = int(self.params['n_prev_match'])
        league_dir = self.params['league_dir']
        update = self.params['update']

        logger.info(f'> Extracting {league_name}')

        # LOADING TRAINING DATA --> ALL DATA SEASON
        league_path = f'{league_dir}{league_name}/{league_name}_npm={n_prev_match}.csv' \
            if league_dir is not None else None

        # LEAGUE CSV ALREADY EXISTING
        print(league_path)
        if league_path is not None and exists(league_path):
            league_df = pd.read_csv(league_path, index_col=0)
            league_df = update_league_data(league_df, n_prev_match) if update else league_df
            league_df = shift_data_features(league_df)
            logger.info('> Updating league data')
            league_df.to_csv(league_path)

        # GENERATING LEAGUE CSV
        else:
            league_df = extract_data(league_name, n_prev_match)
            league_df = shift_data_features(league_df)

            parent = Path(league_path).parent
            if not parent.is_dir():
                os.makedirs(parent, exist_ok=True)
                print(f"Created directory: '{parent}'")
            else:
                print(f"Directory already exists: '{parent}'")
            print(f"Absolute path: {Path(league_path).resolve()}")
            logger.info(f'Saving data at {league_path}')
            league_df.to_csv(league_path)

        return league_df


def update_league_data(league_df, n_prev_match):
    logger.info('> Updating league data')
    league_name = list(league_df['league'].unique())[0]

    assert league_name in LEAGUE.LEAGUE_NAMES, f'Update League Data: Wrong League Name >> {league_name} provided'

    for season_i, path in enumerate(get_league_csv_paths(league_name)):
        season_df = extract_season_data(path, season_i, league_name)

        # ---------CHECK LAST DATE----------
        last_date = pd.to_datetime(league_df.iloc[-1]['Date'])
        date = season_df.iloc[-1]['Date']

        if date > last_date:
            update_df = pd.DataFrame()
            update_df = update_df.append(season_df, sort=False) \
                .reset_index(drop=True)
            update_df = feature_engineering_league(update_df, n_prev_match)
            update_df = update_df[update_df['Date'] > last_date]
            league_df = league_df.append(update_df).reset_index(drop=True)

        # ----------------------------------

    league_df['Date'] = pd.to_datetime(league_df['Date'])

    return league_df