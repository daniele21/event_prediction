shift_features = [
    'cum_home_points', 'cum_away_points',
    'home_league_points', 'away_league_points', 'cum_home_goals', 'cum_away_goals',
    'home_league_goals', 'away_league_goals'
]


def shift_team_features(df, team_type):
    features = [x for x in shift_features if team_type.lower() in x]
    team_df = df[['league', 'season', 'match_n', team_type + 'Team'] + features].copy()
    team_df[features] = team_df.groupby(['league', 'season', team_type + 'Team'])[features].shift(1)
    team_df = team_df.rename(columns={col: f'prev_{col}' for col in features})
    team_df['match_n'] = df['match_n']
    return team_df


def shift_data_features(data):
    # Shift home team features
    home_shifted = shift_team_features(data, 'Home')

    # Shift away team features
    away_shifted = shift_team_features(data, 'Away')

    # Merge the shifted features back into the original DataFrame
    temp_df = home_shifted.merge(away_shifted, on=['league', 'season', 'match_n'], how='left')
    data = data.drop(shift_features, axis=1).copy(deep=True)
    final_data = temp_df.merge(data, on=['league', 'season', 'match_n',
                                         'HomeTeam', 'AwayTeam'], how='left')
    final_data['goals_diff'] = final_data['prev_home_league_goals'] - final_data['prev_away_league_goals']
    final_data['point_diff'] = final_data['prev_home_league_points'] - final_data['prev_away_league_points']

    final_data = final_data[['league', 'season', 'match_n', 'match_day','Date', 'HomeTeam', 'AwayTeam',
                             'home_goals', 'away_goals', 'result_1X2', 'home_points', 'away_points',
                             'bet_1', 'bet_X', 'bet_2', 'point_diff', 'goals_diff',
                             'prev_cum_home_points',
                             'prev_home_league_points', 'prev_cum_home_goals',
                             'prev_home_league_goals', 'prev_cum_away_points',
                             'prev_away_league_points', 'prev_cum_away_goals',
                             'prev_away_league_goals',
                             'HOME_last-1-away', 'HOME_last-1',
                             'AWAY_last-1-home', 'AWAY_last-1', 'AWAY_last-1-away', 'AWAY_last-2',
                             'HOME_last-1-home', 'HOME_last-2', 'HOME_last-2-away', 'HOME_last-3',
                             'AWAY_last-2-home', 'AWAY_last-3', 'HOME_last-2-home', 'HOME_last-4',
                             'AWAY_last-2-away', 'AWAY_last-4', 'HOME_last-3-away', 'HOME_last-5',
                             'AWAY_last-3-home', 'AWAY_last-5', 'HOME_last-3-home',
                             'AWAY_last-3-away', 'HOME_last-4-away', 'AWAY_last-4-home',
                             'HOME_last-4-home', 'AWAY_last-4-away', 'HOME_last-5-away',
                             'AWAY_last-5-home', 'HOME_last-5-home', 'AWAY_last-5-away',
                             ]]
    return final_data
