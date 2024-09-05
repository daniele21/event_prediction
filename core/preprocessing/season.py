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


# def compute_team_stats(df, team_name, match_date, window):
#     # Filter matches where the team was either home or away
#     relevant_matches = df[
#         ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & (df['Date'] < match_date)].copy()
#
#     # Assign points based on whether the team was home or away
#     relevant_matches['Points'] = np.where(
#         relevant_matches['HomeTeam'] == team_name,
#         relevant_matches['FTR'].replace({'H': 3, 'D': 1, 'A': 0}),
#         relevant_matches['FTR'].replace({'A': 3, 'D': 1, 'H': 0})
#     )
#
#     # Assign goals scored and conceded based on whether the team was home or away
#     relevant_matches['GoalsScored'] = np.where(
#         relevant_matches['HomeTeam'] == team_name,
#         relevant_matches['FTHG'],
#         relevant_matches['FTAG']
#     )
#
#     relevant_matches['GoalsConceded'] = np.where(
#         relevant_matches['HomeTeam'] == team_name,
#         relevant_matches['FTAG'],
#         relevant_matches['FTHG']
#     )
#
#     # Separate home and away matches for form calculations
#     home_matches = relevant_matches[relevant_matches['HomeTeam'] == team_name]
#     away_matches = relevant_matches[relevant_matches['AwayTeam'] == team_name]
#
#     # Calculate rolling mean and handle empty series cases for overall form
#     stats = {'OverallPoints': relevant_matches['Points'].rolling(window=window).mean().iloc[-1] if len(
#         relevant_matches['Points']) >= window else None,
#              'GoalsScored': relevant_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(
#                  relevant_matches['GoalsScored']) >= window else None,
#              'GoalsConceded': relevant_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(
#                  relevant_matches['GoalsConceded']) >= window else None, 'GoalDifference':
#                  (relevant_matches['GoalsScored'] - relevant_matches['GoalsConceded']).rolling(
#                      window=window).mean().iloc[
#                      -1] if len(relevant_matches['GoalsScored']) >= window else None,
#              'HomeFormPoints': home_matches['Points'].rolling(window=window).mean().iloc[-1] if len(
#                  home_matches['Points']) >= window else None,
#              'AwayFormPoints': away_matches['Points'].rolling(window=window).mean().iloc[-1] if len(
#                  away_matches['Points']) >= window else None,
#              'HomeGoalsScored': home_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(
#                  home_matches['GoalsScored']) >= window else None,
#              'AwayGoalsScored': away_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(
#                  away_matches['GoalsScored']) >= window else None,
#              'HomeGoalsConceded': home_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(
#                  home_matches['GoalsConceded']) >= window else None,
#              'AwayGoalsConceded': away_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(
#                  away_matches['GoalsConceded']) >= window else None, 'HomeGoalDifference':
#                  (home_matches['GoalsScored'] - home_matches['GoalsConceded']).rolling(window=window).mean().iloc[
#                      -1] if len(
#                      home_matches['GoalsScored']) >= window else None, 'AwayGoalDifference':
#                  (away_matches['GoalsScored'] - away_matches['GoalsConceded']).rolling(window=window).mean().iloc[
#                      -1] if len(
#                      away_matches['GoalsScored']) >= window else None}
#
#     return stats


def compute_team_stats(df, team_name, match_date, window):
    # Filter matches where the team was either home or away
    relevant_matches = df[
        ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & (df['Date'] < match_date)].copy()

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

    # Calculate rolling mean (Moving Average) and Exponential Moving Average (EMA)
    stats = {
        # Moving Averages (MA)
        'OverallPoints_MA': relevant_matches['Points'].rolling(window=window).mean().iloc[-1] if len(
            relevant_matches['Points']) >= window else None,
        'GoalsScored_MA': relevant_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(
            relevant_matches['GoalsScored']) >= window else None,
        'GoalsConceded_MA': relevant_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(
            relevant_matches['GoalsConceded']) >= window else None,
        'GoalDifference_MA': (relevant_matches['GoalsScored'] - relevant_matches['GoalsConceded']).rolling(
            window=window).mean().iloc[-1] if len(relevant_matches['GoalsScored']) >= window else None,

        'HomeFormPoints_MA': home_matches['Points'].rolling(window=window).mean().iloc[-1] if len(
            home_matches['Points']) >= window else None,
        'AwayFormPoints_MA': away_matches['Points'].rolling(window=window).mean().iloc[-1] if len(
            away_matches['Points']) >= window else None,
        'HomeGoalsScored_MA': home_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(
            home_matches['GoalsScored']) >= window else None,
        'AwayGoalsScored_MA': away_matches['GoalsScored'].rolling(window=window).mean().iloc[-1] if len(
            away_matches['GoalsScored']) >= window else None,
        'HomeGoalsConceded_MA': home_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(
            home_matches['GoalsConceded']) >= window else None,
        'AwayGoalsConceded_MA': away_matches['GoalsConceded'].rolling(window=window).mean().iloc[-1] if len(
            away_matches['GoalsConceded']) >= window else None,
        'HomeGoalDifference_MA': (home_matches['GoalsScored'] - home_matches['GoalsConceded']).rolling(
            window=window).mean().iloc[-1] if len(home_matches['GoalsScored']) >= window else None,
        'AwayGoalDifference_MA': (away_matches['GoalsScored'] - away_matches['GoalsConceded']).rolling(
            window=window).mean().iloc[-1] if len(away_matches['GoalsScored']) >= window else None,

        # Exponential Moving Averages (EMA)
        'OverallPoints_EMA': relevant_matches['Points'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            relevant_matches['Points']) > 0 else None,
        'GoalsScored_EMA': relevant_matches['GoalsScored'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            relevant_matches['GoalsScored']) > 0 else None,
        'GoalsConceded_EMA': relevant_matches['GoalsConceded'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            relevant_matches['GoalsConceded']) > 0 else None,
        'GoalDifference_EMA': (relevant_matches['GoalsScored'] - relevant_matches['GoalsConceded']).ewm(
            span=window, adjust=False).mean().iloc[-1] if len(relevant_matches['GoalsScored']) > 0 else None,

        'HomeFormPoints_EMA': home_matches['Points'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            home_matches['Points']) > 0 else None,
        'AwayFormPoints_EMA': away_matches['Points'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            away_matches['Points']) > 0 else None,
        'HomeGoalsScored_EMA': home_matches['GoalsScored'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            home_matches['GoalsScored']) > 0 else None,
        'AwayGoalsScored_EMA': away_matches['GoalsScored'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            away_matches['GoalsScored']) > 0 else None,
        'HomeGoalsConceded_EMA': home_matches['GoalsConceded'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            home_matches['GoalsConceded']) > 0 else None,
        'AwayGoalsConceded_EMA': away_matches['GoalsConceded'].ewm(span=window, adjust=False).mean().iloc[-1] if len(
            away_matches['GoalsConceded']) > 0 else None,
        'HomeGoalDifference_EMA': (home_matches['GoalsScored'] - home_matches['GoalsConceded']).ewm(
            span=window, adjust=False).mean().iloc[-1] if len(home_matches['GoalsScored']) > 0 else None,
        'AwayGoalDifference_EMA': (away_matches['GoalsScored'] - away_matches['GoalsConceded']).ewm(
            span=window, adjust=False).mean().iloc[-1] if len(away_matches['GoalsScored']) > 0 else None
    }

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

            # Adding home team MA stats
            data.at[index, f'HomeTeamOverallForm_MA_{window}'] = home_stats['OverallPoints_MA']
            data.at[index, f'HomeTeamHomeForm_MA_{window}'] = home_stats['HomeFormPoints_MA']
            data.at[index, f'HomeTeamAwayForm_MA_{window}'] = home_stats['AwayFormPoints_MA']
            data.at[index, f'HomeTeamGoalsScored_MA_{window}'] = home_stats['GoalsScored_MA']
            data.at[index, f'HomeTeamGoalsConceded_MA_{window}'] = home_stats['GoalsConceded_MA']
            data.at[index, f'HomeTeamGoalDifference_MA_{window}'] = home_stats['GoalDifference_MA']

            data.at[index, f'HomeTeamHomeGoalsScored_MA_{window}'] = home_stats['HomeGoalsScored_MA']
            data.at[index, f'HomeTeamAwayGoalsScored_MA_{window}'] = home_stats['AwayGoalsScored_MA']
            data.at[index, f'HomeTeamHomeGoalsConceded_MA_{window}'] = home_stats['HomeGoalsConceded_MA']
            data.at[index, f'HomeTeamAwayGoalsConceded_MA_{window}'] = home_stats['AwayGoalsConceded_MA']
            data.at[index, f'HomeTeamHomeGoalDifference_MA_{window}'] = home_stats['HomeGoalDifference_MA']
            data.at[index, f'HomeTeamAwayGoalDifference_MA_{window}'] = home_stats['AwayGoalDifference_MA']

            # Adding home team EMA stats
            data.at[index, f'HomeTeamOverallForm_EMA_{window}'] = home_stats['OverallPoints_EMA']
            data.at[index, f'HomeTeamHomeForm_EMA_{window}'] = home_stats['HomeFormPoints_EMA']
            data.at[index, f'HomeTeamAwayForm_EMA_{window}'] = home_stats['AwayFormPoints_EMA']
            data.at[index, f'HomeTeamGoalsScored_EMA_{window}'] = home_stats['GoalsScored_EMA']
            data.at[index, f'HomeTeamGoalsConceded_EMA_{window}'] = home_stats['GoalsConceded_EMA']
            data.at[index, f'HomeTeamGoalDifference_EMA_{window}'] = home_stats['GoalDifference_EMA']

            data.at[index, f'HomeTeamHomeGoalsScored_EMA_{window}'] = home_stats['HomeGoalsScored_EMA']
            data.at[index, f'HomeTeamAwayGoalsScored_EMA_{window}'] = home_stats['AwayGoalsScored_EMA']
            data.at[index, f'HomeTeamHomeGoalsConceded_EMA_{window}'] = home_stats['HomeGoalsConceded_EMA']
            data.at[index, f'HomeTeamAwayGoalsConceded_EMA_{window}'] = home_stats['AwayGoalsConceded_EMA']
            data.at[index, f'HomeTeamHomeGoalDifference_EMA_{window}'] = home_stats['HomeGoalDifference_EMA']
            data.at[index, f'HomeTeamAwayGoalDifference_EMA_{window}'] = home_stats['AwayGoalDifference_EMA']

            # Adding away team MA stats
            data.at[index, f'AwayTeamOverallForm_MA_{window}'] = away_stats['OverallPoints_MA']
            data.at[index, f'AwayTeamHomeForm_MA_{window}'] = away_stats['HomeFormPoints_MA']
            data.at[index, f'AwayTeamAwayForm_MA_{window}'] = away_stats['AwayFormPoints_MA']
            data.at[index, f'AwayTeamGoalsScored_MA_{window}'] = away_stats['GoalsScored_MA']
            data.at[index, f'AwayTeamGoalsConceded_MA_{window}'] = away_stats['GoalsConceded_MA']
            data.at[index, f'AwayTeamGoalDifference_MA_{window}'] = away_stats['GoalDifference_MA']

            data.at[index, f'AwayTeamHomeGoalsScored_MA_{window}'] = away_stats['HomeGoalsScored_MA']
            data.at[index, f'AwayTeamAwayGoalsScored_MA_{window}'] = away_stats['AwayGoalsScored_MA']
            data.at[index, f'AwayTeamHomeGoalsConceded_MA_{window}'] = away_stats['HomeGoalsConceded_MA']
            data.at[index, f'AwayTeamAwayGoalsConceded_MA_{window}'] = away_stats['AwayGoalsConceded_MA']
            data.at[index, f'AwayTeamHomeGoalDifference_MA_{window}'] = away_stats['HomeGoalDifference_MA']
            data.at[index, f'AwayTeamAwayGoalDifference_MA_{window}'] = away_stats['AwayGoalDifference_MA']

            # Adding away team EMA stats
            data.at[index, f'AwayTeamOverallForm_EMA_{window}'] = away_stats['OverallPoints_EMA']
            data.at[index, f'AwayTeamHomeForm_EMA_{window}'] = away_stats['HomeFormPoints_EMA']
            data.at[index, f'AwayTeamAwayForm_EMA_{window}'] = away_stats['AwayFormPoints_EMA']
            data.at[index, f'AwayTeamGoalsScored_EMA_{window}'] = away_stats['GoalsScored_EMA']
            data.at[index, f'AwayTeamGoalsConceded_EMA_{window}'] = away_stats['GoalsConceded_EMA']
            data.at[index, f'AwayTeamGoalDifference_EMA_{window}'] = away_stats['GoalDifference_EMA']

            data.at[index, f'AwayTeamHomeGoalsScored_EMA_{window}'] = away_stats['HomeGoalsScored_EMA']
            data.at[index, f'AwayTeamAwayGoalsScored_EMA_{window}'] = away_stats['AwayGoalsScored_EMA']
            data.at[index, f'AwayTeamHomeGoalsConceded_EMA_{window}'] = away_stats['HomeGoalsConceded_EMA']
            data.at[index, f'AwayTeamAwayGoalsConceded_EMA_{window}'] = away_stats['AwayGoalsConceded_EMA']
            data.at[index, f'AwayTeamHomeGoalDifference_EMA_{window}'] = away_stats['HomeGoalDifference_EMA']
            data.at[index, f'AwayTeamAwayGoalDifference_EMA_{window}'] = away_stats['AwayGoalDifference_EMA']

    data = create_seasonal_features(data.copy(), windows)

    data = data.rename(columns={'FTHG': 'home_goals',
                                'FTAG': 'away_goals',
                                'FTR': 'result_1X2',
                                'B365H': 'bet_1',
                                'B365D': 'bet_X',
                                'B365A': 'bet_2'})
    data = data[data['result_1X2'].notnull()]
    data['result_1X2'] = data['result_1X2'].apply(encode_result)

    return data


def encode_result(x):
    if str(x) == "H":
        return "1"
    elif str(x) == "A":
        return "2"
    elif str(x) == "D":
        return "X"
    elif str(x) == 'UNKNOWN':
        return str(x)
    else:
        raise AttributeError(f'No match result value found for >> {x} << ')


def create_seasonal_features(data, windows):
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

    data['HomePointsPerMatchDay'] = data['HomeCumulativePoints'] / data['match_day']
    data['AwayPointsPerMatchDay'] = data['AwayCumulativePoints'] / data['match_day']
    data['PointsDifferencePerMatch'] = data['HomePointsPerMatchDay'] - data['AwayPointsPerMatchDay']

    data['TimeSinceSeasonStart'] = data['MatchDay'] / data['MatchDay'].max()
    data['SeasonReset'] = (data['MatchDay'] == 1).astype(int)

    for window in windows:
        data[f'HomeSmoothedPoints_{window}'] = data['HomeCumulativePoints'].ewm(span=window, adjust=False).mean()
        data[f'AwaySmoothedPoints_{window}'] = data['AwayCumulativePoints'].ewm(span=window, adjust=False).mean()
        data[f'SmoothedPointsDifference_{window}'] = data['PointsDifference'].ewm(span=window, adjust=False).mean()

    data = data.drop(['HomeCumulativePoints',
                      'AwayCumulativePoints',
                      'PointsDifference'], axis=1)

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
