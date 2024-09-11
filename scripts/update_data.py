import pandas as pd

import config.league as LEAGUE
from config.data_path import get_league_csv_paths
from core.ingestion.load_data import extract_season_data
from core.ingestion.update_league_data import update_data_league, get_most_recent_data
from core.odds.scraper import load_next_odds
from core.preprocessing.h2h_stats import calculate_h2h_stats


# from core.preprocessing.preprocessing import calculate_h2h_stats


def create_update_league_data(params):
    data = update_data_league(params)

    return data


def get_next_match_day_data(params):
    data = get_most_recent_data(params)

    next_match_day = load_next_odds(params['league_name'])
    next_match_day = next_match_day[next_match_day['bet_1'].notnull()]
    next_match_day_wo_odds = next_match_day.drop(['bookmaker', 'bet_1', 'bet_X', 'bet_2'], axis=1) \
        .drop_duplicates()

    last_season = data['season'].unique()[-1]
    league_paths = get_league_csv_paths(params['league_name'])
    league_path = league_paths[-1]
    season_df = extract_season_data(league_path,
                                    last_season,
                                    params['league_name'],
                                    windows=params['windows'],
                                    next_matches=next_match_day_wo_odds)
    season_df['Date'] = [x.date() for x in season_df['Date']]

    new_data = pd.concat((data[data['season'] < last_season], season_df))
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    new_data = new_data.reset_index(drop=True)

    new_data = calculate_h2h_stats(new_data, params)
    # for window in params['windows']:
    #     new_data[[f'H2H_HomeWinRate_{window}', f'H2H_AwayWinRate_{window}',
    #                f'H2H_GoalDifference_{window}']] = new_data.apply(
    #         lambda row: calculate_h2h_stats(new_data, row['HomeTeam'], row['AwayTeam'], row['Date'],
    #                                         window), axis=1)

    cols = new_data.columns.tolist()
    cols.append('bookmaker')
    next_match_df = new_data[new_data['result_1X2'] == 'UNKNOWN'] \
        .drop(['bet_1', 'bet_X', 'bet_2'], axis=1) \
        .merge(next_match_day.drop('date', axis=1),
               how='left',
               on=['match_day', 'HomeTeam', 'AwayTeam'])
    bookmaker = next_match_df['bookmaker']
    next_match_df = next_match_df[cols]
    next_match_df['bookmaker'] = bookmaker
    new_data['bookmaker'] = None

    final_data = pd.concat((new_data[new_data['result_1X2'] != 'UNKNOWN'],
                            next_match_df[new_data.columns]))\
                .reset_index(drop=True)

    return final_data


if __name__ == '__main__':
    league_name = LEAGUE.SERIE_A
    windows = [1, 3, 5]
    params = {'league_name': league_name,
              'windows': windows,
              'league_dir': f"resources/",
              'update': True}

    data = create_update_league_data(params)
    # get_next_match_day_data(params)
