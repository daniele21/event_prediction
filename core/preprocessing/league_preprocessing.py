import numpy as np
from tqdm import tqdm

from core.logger import logger
from core.preprocessing.preprocessing import _compute_cumultive_feature
from scripts.utils.data_utils import search_previous_matches, compute_outcome_match

def feature_engineering_league(league_df, n_prev_match):
    logger.info('\t\t\t > Feature Engineering for the league')

    league_df = league_df.rename(columns={'B365H': 'bet_1',
                                          'B365D': 'bet_X',
                                          'B365A': 'bet_2',
                                          'FTR': 'result_1X2',
                                          'FTHG': 'home_goals',
                                          'FTAG': 'away_goals'})

    league_df.loc[league_df['result_1X2'] == 'H', 'result_1X2'] = '1'
    league_df.loc[league_df['result_1X2'] == 'D', 'result_1X2'] = 'X'
    league_df.loc[league_df['result_1X2'] == 'A', 'result_1X2'] = '2'

    league_df.loc[league_df['result_1X2'] == '1', 'home_points'] = 3
    league_df.loc[league_df['result_1X2'] == '1', 'away_points'] = 0
    league_df.loc[league_df['result_1X2'] == 'X', 'home_points'] = 1
    league_df.loc[league_df['result_1X2'] == 'X', 'away_points'] = 1
    league_df.loc[league_df['result_1X2'] == '2', 'home_points'] = 0
    league_df.loc[league_df['result_1X2'] == '2', 'away_points'] = 3

    league_df = league_df[['league', 'season', 'match_n', 'match_day', 'Date',
                           'HomeTeam', 'AwayTeam', 'home_goals',
                           'away_goals', 'result_1X2',
                           'bet_1', 'bet_X', 'bet_2',
                           'home_points', 'away_points']]

    league_df = league_df.dropna()
    league_df = league_df.reset_index(drop=True)

    league_df = creating_features(league_df)

    league_df = bind_last_matches(league_df, n_prev_match)

    return league_df

def bind_last_matches(league_df, n_prev_match):
    for i_row in tqdm(range(len(league_df)), desc='Binding Last Matches'):
        row = league_df.iloc[i_row]
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        index = row.name

        league_df = _bind_cum_matches(league_df, home_team, date, n_prev_match, index, home=True)
        league_df = _bind_cum_matches(league_df, away_team, date, n_prev_match, index, home=False)

    return league_df

def _bind_cum_matches(league_df, team, date, n_prev_match, index, home):
    prev_matches = {}

    for home_match in [True, False, None]:
        key = 'home' if home_match else 'away' if home_match == False else 'none'
        prev_matches[key] = search_previous_matches(league_df, team, date, n_prev_match, home_match)

    home_factor = 'HOME' if home else 'AWAY' if home == False else None
    assert home_factor is not None, f'ERROR: bind_matches - Wrong Home Factor'

    for i in range(n_prev_match):
        last_home_col = f'{home_factor}_last-{i + 1}-home'
        last_away_col = f'{home_factor}_last-{i + 1}-away'
        last_match_col = f'{home_factor}_last-{i + 1}'

        prev_home_matches = prev_matches['home'].iloc[:i + 1]
        if len(prev_home_matches) >= i + 1:
            league_df.loc[index, last_home_col] = compute_outcome_match(team, prev_home_matches, home_factor=True)

        prev_away_matches = prev_matches['away'].iloc[:i + 1]
        if len(prev_away_matches) >= i + 1:
            league_df.loc[index, last_away_col] = compute_outcome_match(team, prev_away_matches, home_factor=False)

        prev_none_matches = prev_matches['none'].iloc[:i + 1]
        if len(prev_none_matches) >= i + 1:
            league_df.loc[index, last_match_col] = compute_outcome_match(team, prev_none_matches, home_factor=None)

    return league_df

def creating_features(league_df):
    for season in league_df['season'].unique():
        season_df = league_df[league_df['season'] == season].copy()  # Ensure we work on a copy

        # Inizializza le colonne cum_ e league_points con NaN
        season_df.loc[:, 'cum_home_points'] = np.nan
        season_df.loc[:, 'cum_away_points'] = np.nan
        season_df.loc[:, 'home_league_points'] = np.nan
        season_df.loc[:, 'away_league_points'] = np.nan

        for team in season_df['HomeTeam'].unique():
            season_df = _compute_cumultive_feature(season_df, team, feature_name='points')
            season_df = _compute_cumultive_feature(season_df, team, feature_name='goals')

        league_df.loc[league_df['season'] == season, 'cum_home_points'] = season_df['cum_home_points']
        league_df.loc[league_df['season'] == season, 'cum_away_points'] = season_df['cum_away_points']
        league_df.loc[league_df['season'] == season, 'home_league_points'] = season_df['home_league_points']
        league_df.loc[league_df['season'] == season, 'away_league_points'] = season_df['away_league_points']

        league_df.loc[league_df['season'] == season, 'cum_home_goals'] = season_df['cum_home_goals']
        league_df.loc[league_df['season'] == season, 'cum_away_goals'] = season_df['cum_away_goals']
        league_df.loc[league_df['season'] == season, 'home_league_goals'] = season_df['home_league_goals']
        league_df.loc[league_df['season'] == season, 'away_league_goals'] = season_df['away_league_goals']

    league_df.loc[:, 'point_diff'] = league_df['home_league_points'] - league_df['away_league_points']
    league_df.loc[:, 'goals_diff'] = league_df['home_league_goals'] - league_df['away_league_goals']

    return league_df
