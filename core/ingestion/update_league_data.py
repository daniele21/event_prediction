
from core.dataset_manager.db_manager import DatabaseManager
from core.logger import logger


def update_data_league(params):
    league_name = params['league_name']
    npm = params['n_prev_match']

    league_csv = extract_data_league(params)
    last_date = league_csv.iloc[-1]['Date']

    # logger.info(f'> Updating {league_name} npm={npm} at date {last_date}')
    return league_csv

def extract_data_league(params):
    db_manager = DatabaseManager(params)

    # DATA EXTRACTION
    league_df = db_manager.extract_data_league()

    return league_df