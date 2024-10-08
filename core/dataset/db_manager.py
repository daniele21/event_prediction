from os.path import exists

import pandas as pd

import config.league as LEAGUE
from core.preprocessing.h2h_stats import calculate_h2h_stats

from config.data_path import get_league_csv_paths
from core.ingestion.load_data import extract_data, extract_season_data
from core.logger import logger
# from core.preprocessing.preprocessing import calculate_h2h_stats
from core.time_decorator import timing
from core.utils import get_most_recent_data_path, ensure_folder, get_timestamp


class DatabaseManager:

    def __init__(self, params):
        self.params = params

    @timing
    def extract_data_league(self):
        league_name = self.params['league_name']
        windows = list(self.params['windows'])
        league_dir = self.params['league_dir'] + league_name + '/'
        update = self.params['update'] if 'update' in self.params else True

        logger.info(f'> Extracting {league_name}')

        # LOADING TRAINING DATA --> ALL DATA SEASON
        league_path = get_most_recent_data_path(league_dir, league_name, windows)

        # LEAGUE CSV ALREADY EXISTING
        logger.info(f'League path found: {league_path}')
        if league_path is not None and exists(league_path):
            league_df = pd.read_csv(league_path, index_col=0)
            league_df, update = update_league_data(league_df, windows) if update else league_df
            if update:
                league_df = calculate_h2h_stats(league_df, self.params)
                # for window in windows:
                #     league_df[[f'H2H_HomeWinRate_{window}', f'H2H_AwayWinRate_{window}',
                #                f'H2H_GoalDifference_{window}']] = league_df.apply(
                #         lambda row: calculate_h2h_stats(league_df, row['HomeTeam'], row['AwayTeam'], row['Date'],
                #                                         window), axis=1)

                logger.info('> Updating league data')
                ensure_folder(league_dir)
                league_path = f'{league_dir}{league_name}_{"-".join([str(x) for x in windows])}_{get_timestamp()}.csv'
                league_df.to_csv(league_path)
            else:
                logger.info('> No new data to update')
                return league_df

        # GENERATING LEAGUE CSV
        else:
            league_df = extract_data(league_name, windows)
            league_df = calculate_h2h_stats(league_df, self.params)
            # for window in windows:
            #     league_df[[f'H2H_HomeWinRate_{window}', f'H2H_AwayWinRate_{window}',
            #                f'H2H_GoalDifference_{window}']] = league_df.apply(
            #         lambda row: calculate_h2h_stats(league_df, row['HomeTeam'], row['AwayTeam'], row['Date'],
            #                                         window), axis=1)

            ensure_folder(league_dir)
            league_path = f'{league_dir}{league_name}_{"-".join([str(x) for x in windows])}_{get_timestamp()}.csv'
            logger.info(f'Saving data at {league_path}')
            league_df.to_csv(league_path)

        return league_df


@timing
def update_league_data(league_df, windows):
    logger.info('> Updating league data')
    league_name = list(league_df['league'].unique())[0]
    last_season = league_df['season'].unique()[-1]

    assert league_name in LEAGUE.LEAGUE_NAMES, f'Update League Data: Wrong League Name >> {league_name} provided'

    update = False
    league_paths = get_league_csv_paths(league_name)
    league_seasons = [x.split('/')[-2] for x in league_paths]
    league_path = league_paths[-1]

    season_df = extract_season_data(league_path, last_season, league_name)

    # ---------CHECK LAST DATE----------
    last_date = pd.to_datetime(league_df.iloc[-1]['Date'])
    update_date = season_df.iloc[-1]['Date']

    if str(last_season) != str(league_seasons[-1]) or update_date > last_date:
        update_df = pd.DataFrame()
        update_df = pd.concat((update_df, season_df)).reset_index(drop=True)
        update_df = update_df[update_df['Date'] > last_date]
        league_df = pd.concat((league_df, update_df)).reset_index(drop=True)
        update = True

        # ----------------------------------

    league_df['Date'] = pd.to_datetime(league_df['Date'])

    return league_df, update
