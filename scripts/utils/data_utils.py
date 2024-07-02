# -*- coding: utf-8 -*-
import numpy as np


def search_future_WDL(league_df, team_name, n_curr_round, season):
    _, _, _, f_WDL = search_future_home(league_df, team_name, n_curr_round, season)

    if (f_WDL == 'W' or f_WDL == 'D'):
        f_WD = True
    elif (f_WDL == 'L'):
        f_WD = False
    else:
        raise ValueError('Error search future WDL')

    return f_WDL, f_WD


def search_future_bet_WD(league_df, team_name, n_curr_round, season):
    _, _, f_bet_WD, _ = search_future_home(league_df, team_name, n_curr_round, season)

    return f_bet_WD


def search_future_home(league_df, team_name, n_curr_round, season):
    next_match = search_next_match(league_df, team_name, n_curr_round, season)

    if (next_match is None):
        return np.nan, np.nan

    if (next_match['HomeTeam'].values == team_name):
        f_opponent = next_match['AwayTeam'].values[0]
        f_home = True
        f_bet_WD = 1 / ((1 / next_match['bet_1']) + (1 / next_match['bet_X']))
        f_WDL = convert_result_1X2_to_WDL(next_match['result_1X2'].values[0], f_home)

    elif (next_match['AwayTeam'].values == team_name):
        f_opponent = next_match['HomeTeam'].values[0]
        f_home = False
        f_bet_WD = 1 / ((1 / next_match['bet_2']) + (1 / next_match['bet_X']))
        f_WDL = convert_result_1X2_to_WDL(next_match['result_1X2'].values[0], f_home)

    else:
        raise ValueError('> Error: search_future_home')

    return f_home, f_opponent, f_bet_WD.values[0], f_WDL


def search_future_features(league_df, team_name, match_id, season):
    next_match = search_next_match(league_df, team_name, match_id, season)

    if (next_match is None):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if (next_match['HomeTeam'] == team_name):
        f_opponent = next_match['AwayTeam']
        f_home = True
        f_bet_WD = 1 / ((1 / next_match['bet_1']) + (1 / next_match['bet_X']))
        f_WDL = convert_result_1X2_to_WDL(next_match['result_1X2'], f_home)

    elif (next_match['AwayTeam'] == team_name):
        f_opponent = next_match['HomeTeam']
        f_home = False
        f_bet_WD = 1 / ((1 / next_match['bet_2']) + (1 / next_match['bet_X']))
        f_WDL = convert_result_1X2_to_WDL(next_match['result_1X2'], f_home)

    else:
        raise ValueError('> Error: search_future_feature')

    if (f_WDL == 'W' or f_WDL == 'D'):
        f_WD = True
    elif (f_WDL == 'L'):
        f_WD = False
    else:
        raise ValueError('Error search future WDL')

    return f_home, f_opponent, f_bet_WD, f_WDL, f_WD


def search_future_opponent(league_df, team_name, n_curr_round, season):
    _, f_opponent, _, _ = search_future_home(league_df, team_name, n_curr_round, season)

    return f_opponent


def search_next_match(league_df, team_name, match_id, season):
    for i in range(match_id + 1, len(league_df)):
        row = league_df.iloc[i]

        if (row['HomeTeam'] == team_name or row['AwayTeam'] == team_name):
            return row

    return None


def search_previous_matches(league_df, team_name, date,
                            n_prev_matches, home=None):
    if (home == True):
        home_matches = league_df.loc[league_df['HomeTeam'] == team_name]
        home_matches = home_matches[home_matches['Date'] < date]
        prev_matches = home_matches[-n_prev_matches:]

    elif (home == False):
        away_matches = league_df.loc[league_df['AwayTeam'] == team_name]
        away_matches = away_matches[away_matches['Date'] < date]
        prev_matches = away_matches[-n_prev_matches:]

    elif (home is None):
        matches = league_df.loc[(league_df['HomeTeam'] == team_name) | (league_df['AwayTeam'] == team_name)]
        matches = matches[matches['Date'] < date]
        prev_matches = matches[-n_prev_matches:]

    return prev_matches.sort_index(ascending=False)


def compute_outcome_match(team, prev_match, home_factor):
    if len(prev_match) == 0:
        return np.nan

    else:

        if (home_factor == True):
            points = prev_match['home_points'].sum()
        elif (home_factor == False):
            points = prev_match['away_points'].sum()
        else:
            home_matches = prev_match[prev_match['HomeTeam'] == team]
            points = home_matches['home_points'].sum()
            away_matches = prev_match[prev_match['AwayTeam'] == team]
            points += away_matches['away_points'].sum()

        return points


def convert_result_1X2_to_WDL(result_1X2, home):
    assert home == True or home == False, 'ERROR: convert_result_1X2_to_WDL -> Wrong home value'

    if (home):

        if (str(result_1X2) == '1'):
            outcome = 'W'
        elif (str(result_1X2) == 'X'):
            outcome = 'D'
        elif (str(result_1X2) == '2'):
            outcome = 'L'
        else:
            raise ValueError('Error Convert result 1x2 to wdl')

    else:
        if (str(result_1X2) == '1'):
            outcome = 'L'
        elif (str(result_1X2) == 'X'):
            outcome = 'D'
        elif (str(result_1X2) == '2'):
            outcome = 'W'
        else:
            raise ValueError('Error Convert result 1x2 to wdl')

    return outcome


def compute_outcome_points(result_1X2, home):
    home_result = {'1': 3,
                   'X': 1,
                   '2': 0}

    away_result = {'1': 0,
                   'X': 1,
                   '2': 3}

    if (home):
        return home_result[str(result_1X2)]

    else:
        return away_result[str(result_1X2)]


