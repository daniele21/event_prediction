import numpy as np
import pandas as pd


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

def preprocessing_season(season_df, n_season, league_name):
    data = season_df.copy(deep=True)

    data.insert(0, 'season', n_season)
    data.insert(0, 'league', league_name)
    data.insert(2, 'match_n', np.arange(1, len(data) + 1, 1))
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = calculate_team_based_match_day(data)

    return data
