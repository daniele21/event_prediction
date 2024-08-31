import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from core.time_decorator import timing


@timing
def calculate_h2h_stats(df, params):
    home_teams = df['HomeTeam'].values
    away_teams = df['AwayTeam'].values
    dates = pd.to_datetime(df['Date']).values  # Ensure dates are in a NumPy datetime64 format
    results_1X2 = df['result_1X2'].values
    home_goals_array = df['home_goals'].values
    away_goals_array = df['away_goals'].values

    def calculate_h2h_stats_for_row(home_team, away_team, match_date, window):
        # Filter H2H matches up to the match date using numpy boolean indexing
        mask = (((home_teams == home_team) & (away_teams == away_team)) |
                ((home_teams == away_team) & (away_teams == home_team))) & (dates < match_date)
        h2h_matches_indices = np.where(mask)[0][-window:]  # Get indices of the last `window` matches

        if h2h_matches_indices.size == 0:
            return np.nan, np.nan, np.nan

        # Compute stats using numpy indexing
        relevant_results = results_1X2[h2h_matches_indices]
        relevant_home_teams = home_teams[h2h_matches_indices]
        home_wins = ((relevant_home_teams == home_team) & (relevant_results == '1')).sum()
        away_wins = ((relevant_home_teams == away_team) & (relevant_results == '2')).sum()
        draws = (relevant_results == 'X').sum()
        total_matches = len(h2h_matches_indices)

        home_win_rate = (home_wins + draws * 0.5) / total_matches
        away_win_rate = (away_wins + draws * 0.5) / total_matches

        # Goal difference calculation using numpy indexing
        relevant_home_goals = np.where(relevant_home_teams == home_team, home_goals_array[h2h_matches_indices],
                                       away_goals_array[h2h_matches_indices])
        relevant_away_goals = np.where(relevant_home_teams == home_team, away_goals_array[h2h_matches_indices],
                                       home_goals_array[h2h_matches_indices])
        goal_difference = (relevant_home_goals - relevant_away_goals).mean()

        return home_win_rate, away_win_rate, goal_difference

    # Initialize the new columns with NaNs
    for window in params['windows']:
        df[f'H2H_HomeWinRate_{window}'] = np.nan
        df[f'H2H_AwayWinRate_{window}'] = np.nan
        df[f'H2H_GoalDifference_{window}'] = np.nan

    # Parallel computation of H2H stats for each row and window
    for window in params['windows']:
        results = Parallel(n_jobs=-1, backend='loky')(delayed(calculate_h2h_stats_for_row)(
            home_team, away_team, match_date, window
        ) for home_team, away_team, match_date in zip(home_teams, away_teams, dates))

        # Assign computed values to the corresponding columns in the original DataFrame
        df[f'H2H_HomeWinRate_{window}'] = [res[0] for res in results]
        df[f'H2H_AwayWinRate_{window}'] = [res[1] for res in results]
        df[f'H2H_GoalDifference_{window}'] = [res[2] for res in results]

    return df
