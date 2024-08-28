import pandas as pd

from core.dataset_manager.db_manager import DatabaseManager
from core.logger import logger
from core.utils import get_most_recent_data_path


def get_most_recent_data(dataset_params):
    league_name = dataset_params['league_name']
    league_dir = dataset_params['league_dir'] + league_name + '/'
    windows = dataset_params['windows']

    league_path = get_most_recent_data_path(league_dir, league_name, windows)
    if league_path:
        return pd.read_csv(league_path, index_col=0)
    else:
        return update_data_league(dataset_params)

def update_data_league(params):

    league_csv = extract_data_league(params)
    last_date = league_csv.iloc[-1]['Date']

    # logger.info(f'> Updating {league_name} npm={npm} at date {last_date}')
    return league_csv

def extract_data_league(params):
    db_manager = DatabaseManager(params)

    # DATA EXTRACTION
    league_df = db_manager.extract_data_league()

    return league_df