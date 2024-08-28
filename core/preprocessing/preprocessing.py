# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from math import ceil
import os
from tqdm import tqdm

from core.logger import logger
# from core.str2bool import str2bool
# from scripts.data import constants as K
# from scripts.data.data_utils import (search_previous_matches,
#                                      search_future_features,
#                                      search_future_WDL,
#                                      search_future_opponent,
#                                      search_future_home,
#                                      search_future_bet_WD,
#                                      compute_outcome_match,
#                                      convert_result_1X2_to_WDL, compute_outcome_points)
# from core.logger.logging import logger
# from core.file_manager.os_utils import exists
from core.time_decorator import timing
from multiprocessing import Pool
from functools import partial

def calculate_h2h_stats(df, home_team, away_team, match_date, window=5):
    h2h_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                     ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
    h2h_matches = h2h_matches[h2h_matches['Date'] < match_date].tail(window)

    if h2h_matches.empty:
        return pd.Series([np.nan, np.nan, np.nan], index=['H2H_HomeWinRate', 'H2H_AwayWinRate', 'H2H_GoalDifference'])

    # Calculate win rates for both home and away perspectives
    home_wins = len(h2h_matches[(h2h_matches['HomeTeam'] == home_team) & (h2h_matches['result_1X2'] == '1')])
    away_wins = len(h2h_matches[(h2h_matches['AwayTeam'] == away_team) & (h2h_matches['result_1X2'] == '2')])
    draws = len(h2h_matches[h2h_matches['result_1X2'] == 'X'])

    total_matches = len(h2h_matches)

    home_win_rate = (home_wins + draws * 0.5) / total_matches
    away_win_rate = (away_wins + draws * 0.5) / total_matches

    # Goal difference calculation (home goals - away goals)
    home_goals = h2h_matches.apply(lambda row: row['home_goals'] if row['HomeTeam'] == home_team else row['away_goals'], axis=1)
    away_goals = h2h_matches.apply(lambda row: row['away_goals'] if row['HomeTeam'] == home_team else row['home_goals'], axis=1)

    goal_difference = (home_goals - away_goals).mean()

    return pd.Series([home_win_rate, away_win_rate, goal_difference],
                     index=['H2H_HomeWinRate', 'H2H_AwayWinRate', 'H2H_GoalDifference'])


def _addRound(season_csv):
    data = season_csv.copy(deep=True)
    n_teams = len(K.SERIE_A_TEAMS)

    rounds = []

    for i in range(len(data)):
        index = data.iloc[i]['match_n']
        n_round = ceil(2 * index / n_teams)
        rounds.append(n_round)

    data.insert(3, 'round', rounds)

    return data


def bind_last_matches(league_df, n_prev_match):
    for i_row in tqdm(range(len(league_df)), desc='Binding Last Matches'):
        row = league_df.iloc[i_row]
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        index = row.name

        # league_df = _bind_matches(league_df, home_team, date, n_prev_match, index, home=True)
        # league_df = _bind_matches(league_df, away_team, date, n_prev_match, index, home=False)

        league_df = _bind_cum_matches(league_df, home_team, date, n_prev_match, index, home=True)
        league_df = _bind_cum_matches(league_df, away_team, date, n_prev_match, index, home=False)

    return league_df


def _bind_matches(league_df, team, date, n_prev_match, index, home):
    prev_matches = {}

    for home_match in [True, False, None]:
        key = 'home' if home_match == True else 'away' if home_match == False else 'none'
        prev_matches[key] = search_previous_matches(league_df, team, date, n_prev_match, home_match)

    home_factor = 'HOME' if home == True else 'AWAY' if home == False else None
    assert home_factor is not None, f'ERROR: bind_matches - Wrong Home Factor'

    for i in range(0, n_prev_match):
        last_home_col = f'{home_factor}_last-{i + 1}-home'
        last_away_col = f'{home_factor}_last-{i + 1}-away'
        last_match_col = f'{home_factor}_last-{i + 1}'

        league_df.loc[index, last_home_col] = compute_outcome_match(team, prev_matches['home'], i)
        league_df.loc[index, last_away_col] = compute_outcome_match(team, prev_matches['away'], i)
        league_df.loc[index, last_match_col] = compute_outcome_match(team, prev_matches['none'], i)

    return league_df





def _bind_trend_last_previous_match(league_df, n_prev_match):
    for i_row in range(len(league_df)):
        row = league_df.iloc[i_row]
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        #        season = row['season']
        date = row['Date']
        index = row.name

        # HOME TEAM
        home_prev_matches = search_previous_matches(league_df, home_team, date, n_prev_match, home=True)
        away_prev_matches = search_previous_matches(league_df, home_team, date, n_prev_match, home=False)
        prev_matches = search_previous_matches(league_df, home_team, date, n_prev_match, home=None)

        for i in range(0, n_prev_match):
            last_home_col = 'HOME_last-{}-home'.format(i + 1)
            last_away_col = 'HOME_last-{}-away'.format(i + 1)
            last_match_col = 'HOME_last-{}'.format(i + 1)

            league_df.loc[index, last_home_col] = compute_outcome_match(home_team, home_prev_matches, i)
            league_df.loc[index, last_away_col] = compute_outcome_match(home_team, away_prev_matches, i)
            league_df.loc[index, last_match_col] = compute_outcome_match(home_team, prev_matches, i)

        # AWAY TEAM
        home_prev_matches = search_previous_matches(league_df, away_team, date, n_prev_match, home=True)
        away_prev_matches = search_previous_matches(league_df, away_team, date, n_prev_match, home=False)
        prev_matches = search_previous_matches(league_df, away_team, date, n_prev_match, home=None)

        for i in range(0, n_prev_match):
            last_home_col = 'AWAY_last-{}-home'.format(i + 1)
            last_away_col = 'AWAY_last-{}-away'.format(i + 1)
            last_match_col = 'AWAY_last-{}'.format(i + 1)

            league_df.loc[index, last_home_col] = compute_outcome_match(away_team, home_prev_matches, i)
            league_df.loc[index, last_away_col] = compute_outcome_match(away_team, away_prev_matches, i)
            league_df.loc[index, last_match_col] = compute_outcome_match(away_team, prev_matches, i)

    return league_df


def _compute_cumultive_feature(season_df, team, feature_name):
    home_team_df = season_df[season_df['HomeTeam'] == team]
    away_team_df = season_df[season_df['AwayTeam'] == team]

    home_cum_feature = home_team_df[f'home_{feature_name}'].cumsum()
    away_cum_feature = away_team_df[f'away_{feature_name}'].cumsum()

    season_df.loc[home_team_df.index, f'overall_home'] = home_team_df[f'home_{feature_name}']
    season_df.loc[away_team_df.index, f'overall_away'] = away_team_df[f'away_{feature_name}']
    season_df.loc[home_team_df.index, f'cum_home_{feature_name}'] = home_cum_feature
    season_df.loc[away_team_df.index, f'cum_away_{feature_name}'] = away_cum_feature

    team_df = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)]

    overall_df = team_df[['overall_home', 'overall_away']].copy(deep=True)
    overall_df['overall_home'] = overall_df['overall_home'].fillna(overall_df['overall_away'])
    overall_list = overall_df['overall_home'].cumsum().to_list()

    season_df = season_df.drop('overall_home', axis=1)
    season_df = season_df.drop('overall_away', axis=1)

    for i, index in enumerate(team_df.index):
        row = team_df.loc[index]
        home = True if row['HomeTeam'] == team else False

        if home:
            season_df.loc[index, f'home_league_{feature_name}'] = overall_list[i]
        else:
            season_df.loc[index, f'away_league_{feature_name}'] = overall_list[i]

    return season_df


def _compute_cumultive_points(season_df, team):
    home_team_df = season_df[season_df['HomeTeam'] == team]
    away_team_df = season_df[season_df['AwayTeam'] == team]

    home_cum_points = home_team_df['home_points'].cumsum()
    away_cum_points = away_team_df['away_points'].cumsum()

    season_df.loc[home_team_df.index, 'cum_points_H'] = home_team_df['home_reward']
    season_df.loc[away_team_df.index, 'cum_points_A'] = away_team_df['away_reward']
    season_df.loc[home_team_df.index, 'cum_home_points'] = home_cum_points
    season_df.loc[away_team_df.index, 'cum_away_points'] = away_cum_points

    team_df = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)]

    cum_points_df = team_df[['cum_points_H', 'cum_points_A']].copy(deep=True)
    cum_points_df['cum_points_H'] = cum_points_df['cum_points_H'].fillna(cum_points_df['cum_points_A'])
    cum_points = cum_points_df['cum_points_H'].cumsum().to_list()
    season_df = season_df.drop('cum_points_H', axis=1)
    season_df = season_df.drop('cum_points_A', axis=1)

    for i, index in enumerate(team_df.index):
        row = team_df.loc[index]
        home = True if row['HomeTeam'] == team else False

        if home:
            season_df.loc[index, 'home_league_points'] = cum_points[i]
        else:
            season_df.loc[index, 'away_league_points'] = cum_points[i]

    return season_df








def _split_teams_one_row(data, i_row, n_prev_match, home):
    row = data.iloc[i_row]

    field = 'home' if home else 'away'
    team = row['HomeTeam'] if home else row['AwayTeam']
    opponent = row['AwayTeam'] if home else row['HomeTeam']
    date = row['Date']
    goal_scored = row['home_goals'] if home else row['away_goals']
    goal_conceded = row['away_goals'] if home else row['home_goals']
    # result_1X2 = row['result_1X2']
    season = row['season']
    league = row['league']
    points = row['home_points'] if home else row['away_points']
    cum_points = row['cum_home_points'] if home else row['cum_away_points']
    league_points = row['home_league_points'] if home else row['away_league_points']
    cum_goals = row['cum_home_goals'] if home else row['cum_away_goals']
    league_goals = row['home_league_goals'] if home else row['away_league_goals']
    point_diff = row['point_diff'] if home else (row['point_diff'] * -1)
    goals_diff = row['goals_diff'] if home else (row['goals_diff'] * -1)

    f_home, f_opponent, f_bet_WD, f_WDL, f_WD = search_future_features(data, team, i_row, season)

    team_features = {'league': league,
                     'season': season,
                     'team': team,
                     'opponent': opponent,
                     'date': date,
                     'goal_scored': goal_scored,
                     'goal_conceded': goal_conceded,
                     'points': points,
                     'home': home,
                     'f-opponent': f_opponent,
                     'f-home': f_home,
                     'f-bet-WD': f_bet_WD,
                     'f-result-WDL': f_WDL,
                     'f-WD': f_WD,
                     f'cum_{field}_points': cum_points,
                     f'cum_{field}_goals': cum_goals,
                     'league_points': league_points,
                     'league_goals': league_goals,
                     'point_diff': point_diff,
                     'goals_diff': goals_diff
                     }

    team_features['bet-WD'] = 1 / ((1 / row['bet_1']) + (1 / row['bet_X']))

    home_factor = 'HOME' if home else 'AWAY'
    for n in range(1, n_prev_match + 1):
        team_features[f'last-{n}_home'] = row[f'{home_factor}_last-{n}-home']
        team_features[f'last-{n}_home'] = row[f'{home_factor}_last-{n}-home']
        team_features[f'last-{n}_home'] = row[f'{home_factor}_last-{n}-home']

        team_features[f'last-{n}_away'] = row[f'{home_factor}_last-{n}-away']
        team_features[f'last-{n}_away'] = row[f'{home_factor}_last-{n}-away']
        team_features[f'last-{n}_away'] = row[f'{home_factor}_last-{n}-away']

        team_features[f'last-{n}'] = row[f'{home_factor}_last-{n}']
        team_features[f'last-{n}'] = row[f'{home_factor}_last-{n}']
        team_features[f'last-{n}'] = row[f'{home_factor}_last-{n}']

    return team_features


def _split_teams(league_df, n_prev_match):
    """
    Splitting matches data into home data and away data

    Args:
        league_df:
        n_prev_match:

    Returns:

    """
    data = league_df.copy(deep=True)
    home_team_df = pd.DataFrame()
    away_team_df = pd.DataFrame()

    for i_row in tqdm(range(len(data)), desc='Splitting teams'):
        home_team_dict = _split_teams_one_row(data, i_row, n_prev_match, home=True)
        away_team_dict = _split_teams_one_row(data, i_row, n_prev_match, home=False)

        home_team_df = home_team_df.append(pd.DataFrame(home_team_dict, index=[i_row]))
        away_team_df = away_team_df.append(pd.DataFrame(away_team_dict, index=[i_row]))

    return {'home': home_team_df,
            'away': away_team_df}


@timing
def data_preprocessing(league_df, params):
    n_prev_match = int(params['n_prev_match'])
    league_dir = params['league_dir']
    league_name = params['league_name']
    update = params['update']

    data = league_df.copy(deep=True)

    input_data = {}
    prep_league_path = {x: f'{league_dir}{league_name}/prep_{x}_{league_name}_npm={n_prev_match}.csv' for x in
                        ['home', 'away']}
    if (league_dir is not None and
            exists(prep_league_path['home']) and
            exists(prep_league_path['away'])):

        for x in ['home', 'away']:
            input_data[x] = pd.read_csv(prep_league_path[x], index_col=0)

        input_data = update_input_data(data, input_data, n_prev_match) if update else input_data

    if len(input_data) == 0:
        input_data = _split_teams(data, n_prev_match)

    if train:
        input_data['home'].to_csv(prep_league_path['home'])
        input_data['away'].to_csv(prep_league_path['away'])
    else:
        input_data['home'] = input_data['home'].iloc[-test_size:]
        input_data['away'] = input_data['away'].iloc[-test_size:]

    return input_data


def update_input_data(league_df, input_data, n_prev_match):
    logger.info(f'> Updating league data preprocessed: data ')
    last_date = pd.to_datetime(league_df.iloc[-1]['Date'])

    date_home = pd.to_datetime(input_data['home'].iloc[-1]['date'])
    date_away = pd.to_datetime(input_data['away'].iloc[-1]['date'])

    assert date_home == date_away
    date = date_home

    if date < last_date:
        data = league_df[league_df['Date'] > date]
        update_data = _split_teams(data, n_prev_match)

        input_data['home'] = input_data['home'].append(update_data['home']).reset_index(drop=True)
        input_data['away'] = input_data['away'].append(update_data['away']).reset_index(drop=True)

    return input_data


def get_last_round(test_data):
    indexes = test_data.index.to_list()

    gaps = [[s, e] for s, e in zip(indexes, indexes[1:]) if s + 1 < e]
    edges = iter(indexes[:1] + sum(gaps, []) + indexes[-1:])
    cons_list = list(zip(edges, edges))

    last_round = cons_list[-1]

    last_round_df = test_data.loc[last_round[0]: last_round[1]]

    return last_round_df


def fill_inference_matches(test_data, matches_dict):
    home, away = matches_dict['home_teams'], matches_dict['away_teams']
    odds_1x, odds_x2 = matches_dict['1X_odds'], matches_dict['X2_odds'],

    home_matches = _fill_per_field(test_data, home, away, odds_1x, f_home=1)
    away_matches = _fill_per_field(test_data, away, home, odds_x2, f_home=0)

    matches = {}
    for field in ['home', 'away']:
        matches[field] = home_matches[field].append(away_matches[field])

    return matches


def _fill_per_field(test_data, field_teams, opponent_teams, odds, f_home):
    matches = {'home': pd.DataFrame(),
               'away': pd.DataFrame()}

    for i, team in enumerate(field_teams):
        for field in ['home', 'away']:
            data = test_data[field]
            team_df = data[data['team'] == team].iloc[-1:]

            if (len(team_df['f-opponent'].isnull() > 0)):
                idx = team_df.index
                opponent = opponent_teams[i]
                odd = odds[i]

                data.loc[idx, 'f-opponent'] = opponent
                data.loc[idx, 'f-home'] = f_home
                data.loc[idx, 'f-bet-WD'] = odd

                matches[field] = matches[field].append(data.loc[idx])

    return matches


def poisson_data_preprocessing(league_df):
    home_columns = {'home_goals': 'goals',
                    'HomeTeam': 'team',
                    'AwayTeam': 'opponent'}

    away_columns = {'away_goals': 'goals',
                    'AwayTeam': 'team',
                    'HomeTeam': 'opponent'}

    home_data = league_df.rename(columns=home_columns).assign(home=1)
    away_data = league_df.rename(columns=away_columns).assign(home=0)

    # poisson_data = pd.concat(home_data, away_data)
    poisson_data = home_data.append(away_data)

    return poisson_data
