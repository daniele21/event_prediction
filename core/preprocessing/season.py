import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_team_based_match_day(df):
    df = df.sort_values(by=['season', 'match_n'])
    df['match_day'] = 0

    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        team_matches = {}

        for idx in season_df.index:
            home_team = df.loc[idx, 'HomeTeam']
            away_team = df.loc[idx, 'AwayTeam']

            if home_team not in team_matches:
                team_matches[home_team] = 0
            if away_team not in team_matches:
                team_matches[away_team] = 0

            team_matches[home_team] += 1
            team_matches[away_team] += 1

            df.at[idx, 'match_day'] = max(team_matches[home_team], team_matches[away_team])

    return df


def compute_team_stats(df, team_name, match_date, window):
    # Filter matches where the team was either home or away
    relevant_matches = df[((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & (df['Date'] < match_date)].copy()

    # Assign points based on whether the team was home or away
    relevant_matches['Points'] = np.where(
        relevant_matches['HomeTeam'] == team_name,
        relevant_matches['FTR'].replace({'H': 3, 'D': 1, 'A': 0}),
        relevant_matches['FTR'].replace({'A': 3, 'D': 1, 'H': 0})
    )

    # Assign goals scored and conceded based on whether the team was home or away
    relevant_matches['GoalsScored'] = np.where(
        relevant_matches['HomeTeam'] == team_name,
        relevant_matches['FTHG'],
        relevant_matches['FTAG']
    )

    relevant_matches['GoalsConceded'] = np.where(
        relevant_matches['HomeTeam'] == team_name,
        relevant_matches['FTAG'],
        relevant_matches['FTHG']
    )

    # Separate home and away matches for form calculations
    home_matches = relevant_matches[relevant_matches['HomeTeam'] == team_name]
    away_matches = relevant_matches[relevant_matches['AwayTeam'] == team_name]

    # Calculate rolling mean and handle empty series cases for overall form
    stats = {
        'OverallPoints': relevant_matches['Points'].rolling(window=window).mean().iloc[-1] if len(relevant_matches['Points']) >= window else None,
        'GoalsScored': relevant_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(relevant_matches['GoalsScored']) >= window else None,
        'GoalsConceded': relevant_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(relevant_matches['GoalsConceded']) >= window else None,
        'GoalDifference': (relevant_matches['GoalsScored'] - relevant_matches['GoalsConceded']).rolling(window=window).mean().iloc[-1] if len(relevant_matches['GoalsScored']) >= window else None
    }

    # Calculate home and away form separately
    stats['HomeFormPoints'] = home_matches['Points'].rolling(window=window).mean().iloc[-1] if len(home_matches['Points']) >= window else None
    stats['AwayFormPoints'] = away_matches['Points'].rolling(window=window).mean().iloc[-1] if len(away_matches['Points']) >= window else None

    # Calculate home and away goal stats separately
    stats['HomeGoalsScored'] = home_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(home_matches['GoalsScored']) >= window else None
    stats['AwayGoalsScored'] = away_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(away_matches['GoalsScored']) >= window else None

    stats['HomeGoalsConceded'] = home_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(home_matches['GoalsConceded']) >= window else None
    stats['AwayGoalsConceded'] = away_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(away_matches['GoalsConceded']) >= window else None

    stats['HomeGoalDifference'] = (home_matches['GoalsScored'] - home_matches['GoalsConceded']).rolling(window=window).mean().iloc[-1] if len(home_matches['GoalsScored']) >= window else None
    stats['AwayGoalDifference'] = (away_matches['GoalsScored'] - away_matches['GoalsConceded']).rolling(window=window).mean().iloc[-1] if len(away_matches['GoalsScored']) >= window else None

    return stats

def preprocessing_season_optimized(season_df, season, league_name, windows=None):
    if windows is None:
        windows = [3, 5, 10]

    data = season_df.copy(deep=True)
    data.insert(0, 'season', season)
    data.insert(0, 'league', league_name)
    data.insert(2, 'match_n', np.arange(1, len(data) + 1, 1))
    data = calculate_team_based_match_day(data)

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    data = data[['league', 'season', 'match_n', 'match_day', 'Date',
                 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                 'FTR', 'B365H', 'B365D', 'B365A']]

    for window in windows:
        for index, row in data.iterrows():
            home_stats = compute_team_stats(data, row['HomeTeam'], row['Date'], window)
            away_stats = compute_team_stats(data, row['AwayTeam'], row['Date'], window)

            # Adding home team stats
            data.at[index, f'HomeTeamOverallForm_{window}'] = home_stats['OverallPoints']
            data.at[index, f'HomeTeamHomeForm_{window}'] = home_stats['HomeFormPoints']
            data.at[index, f'HomeTeamAwayForm_{window}'] = home_stats['AwayFormPoints']
            data.at[index, f'HomeTeamGoalsScored_{window}'] = home_stats['GoalsScored']
            data.at[index, f'HomeTeamGoalsConceded_{window}'] = home_stats['GoalsConceded']
            data.at[index, f'HomeTeamGoalDifference_{window}'] = home_stats['GoalDifference']

            data.at[index, f'HomeTeamHomeGoalsScored_{window}'] = home_stats['HomeGoalsScored']
            data.at[index, f'HomeTeamAwayGoalsScored_{window}'] = home_stats['AwayGoalsScored']
            data.at[index, f'HomeTeamHomeGoalsConceded_{window}'] = home_stats['HomeGoalsConceded']
            data.at[index, f'HomeTeamAwayGoalsConceded_{window}'] = home_stats['AwayGoalsConceded']
            data.at[index, f'HomeTeamHomeGoalDifference_{window}'] = home_stats['HomeGoalDifference']
            data.at[index, f'HomeTeamAwayGoalDifference_{window}'] = home_stats['AwayGoalDifference']

            # Adding away team stats
            data.at[index, f'AwayTeamOverallForm_{window}'] = away_stats['OverallPoints']
            data.at[index, f'AwayTeamHomeForm_{window}'] = away_stats['HomeFormPoints']
            data.at[index, f'AwayTeamAwayForm_{window}'] = away_stats['AwayFormPoints']
            data.at[index, f'AwayTeamGoalsScored_{window}'] = away_stats['GoalsScored']
            data.at[index, f'AwayTeamGoalsConceded_{window}'] = away_stats['GoalsConceded']
            data.at[index, f'AwayTeamGoalDifference_{window}'] = away_stats['GoalDifference']

            data.at[index, f'AwayTeamHomeGoalsScored_{window}'] = away_stats['HomeGoalsScored']
            data.at[index, f'AwayTeamAwayGoalsScored_{window}'] = away_stats['AwayGoalsScored']
            data.at[index, f'AwayTeamHomeGoalsConceded_{window}'] = away_stats['HomeGoalsConceded']
            data.at[index, f'AwayTeamAwayGoalsConceded_{window}'] = away_stats['AwayGoalsConceded']
            data.at[index, f'AwayTeamHomeGoalDifference_{window}'] = away_stats['HomeGoalDifference']
            data.at[index, f'AwayTeamAwayGoalDifference_{window}'] = away_stats['AwayGoalDifference']

    data = create_seasonal_features(data.copy())

    return data


def create_seasonal_features(data):
    """
    This function takes a DataFrame containing match data and adds three new features:
    - HomeCumulativePoints: The cumulative points of the home team up to each match.
    - AwayCumulativePoints: The cumulative points of the away team up to each match.
    - PointsDifference: The difference in cumulative points between the home and away teams.

    The DataFrame should include columns 'HomeTeam', 'AwayTeam', and 'FTR' (Full-Time Result).

    Parameters:
    data (DataFrame): The input match data.

    Returns:
    DataFrame: The DataFrame with the new features added.
    """
    # Initialize dictionaries to keep track of cumulative points for each team
    team_points = {team: 0 for team in pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()}

    # Lists to store the cumulative points and point differences
    home_cumulative_points = []
    away_cumulative_points = []
    points_differences = []

    # Iterate through the matches to calculate cumulative points
    for index, row in data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Store current cumulative points before the match
        home_cumulative_points.append(team_points[home_team])
        away_cumulative_points.append(team_points[away_team])
        # Calculate the difference in cumulative points
        points_differences.append(team_points[home_team] - team_points[away_team])

        # Calculate points based on the match result
        if row['FTR'] == 'H':
            team_points[home_team] += 3
        elif row['FTR'] == 'A':
            team_points[away_team] += 3
        elif row['FTR'] == 'D':
            team_points[home_team] += 1
            team_points[away_team] += 1

    # Add the new features to the dataset
    data['HomeCumulativePoints'] = home_cumulative_points
    data['AwayCumulativePoints'] = away_cumulative_points
    data['PointsDifference'] = points_differences

    return data


# def preprocessing_season(season_df, n_season, league_name):
#     data = season_df.copy(deep=True)
#
#     data.insert(0, 'season', n_season)
#     data.insert(0, 'league', league_name)
#     data.insert(2, 'match_n', np.arange(1, len(data) + 1, 1))
#     try:
#         data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
#     except ValueError as exc:
#         data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
#     data = calculate_team_based_match_day(data)
#
#     return data

