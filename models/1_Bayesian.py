import pandas as pd
import os
import json

features = [
    'Unnamed: 0',
    'league',
    'season',
    'match_n',
    'Date',
    'HomeTeam',
    'AwayTeam',
    'home_goals',
    'away_goals',
    'result_1X2',
    'bet_1',
    'bet_X',
    'bet_2',
    'home_points',
    'away_points',
    'cum_home_points',
    'cum_away_points',
    'home_league_points',
    'away_league_points',
    'cum_home_goals',
    'cum_away_goals',
    'home_league_goals',
    'away_league_goals',
    'point_diff',
    'goals_diff',
    'HOME_last-1-away',
    'HOME_last-1',
    'AWAY_last-1-home',
    'AWAY_last-1',
    'AWAY_last-1-away',
    'AWAY_last-2',
    'HOME_last-1-home',
    'HOME_last-2',
    'HOME_last-2-away',
    'HOME_last-3',
    'AWAY_last-2-home',
    'AWAY_last-3',
    'HOME_last-2-home',
    'HOME_last-4',
    'AWAY_last-2-away',
    'AWAY_last-4',
    'HOME_last-3-away',
    'HOME_last-5',
    'AWAY_last-3-home',
    'AWAY_last-5',
    'HOME_last-3-home',
    'AWAY_last-3-away',
    'HOME_last-4-away',
    'AWAY_last-4-home',
    'HOME_last-4-home',
    'AWAY_last-4-away',
    'HOME_last-5-away',
    'AWAY_last-5-home',
    'HOME_last-5-home',
    'AWAY_last-5-away'
]

class TeamProbability:
    def __init__(self, team_stats, alpha=1.2, beta=0.8):
        # Se team_stats è una stringa, converti in dizionario
        if isinstance(team_stats, str):
            self.teams = json.loads(team_stats)
        else:
            self.teams = team_stats
        self.alpha = alpha
        self.beta = beta

    def calculate_probabilities(self):
        for team, stats in self.teams.items():
            home_wins = stats['home']['wins']
            home_draws = stats['home']['draws']
            home_tot = stats['home']['tot_match']
            away_wins = stats['away']['wins']
            away_draws = stats['away']['draws']
            away_tot = stats['away']['tot_match']

            stats['home_win_prob'] = home_wins / home_tot if home_tot > 0 else 0
            stats['home_draw_prob'] = home_draws / home_tot if home_tot > 0 else 0
            stats['home_loss_prob'] = 1 - stats['home_win_prob'] - stats['home_draw_prob']

            stats['away_win_prob'] = away_wins / away_tot if away_tot > 0 else 0
            stats['away_draw_prob'] = away_draws / away_tot if away_tot > 0 else 0
            stats['away_loss_prob'] = 1 - stats['away_win_prob'] - stats['away_draw_prob']

    def conditional_probabilities(self, teamA, teamB):
        self.calculate_probabilities()

        if teamA not in self.teams or teamB not in self.teams:
            raise ValueError("One or both teams are not in the dataset")

        # Probabilità di vittoria, pareggio e sconfitta per la squadra A contro la squadra B
        P_W_home_A = self.teams[teamA]['home_win_prob']
        P_D_home_A = self.teams[teamA]['home_draw_prob']
        P_L_home_A = self.teams[teamA]['home_loss_prob']

        P_W_away_A = self.teams[teamA]['away_win_prob']
        P_D_away_A = self.teams[teamA]['away_draw_prob']
        P_L_away_A = self.teams[teamA]['away_loss_prob']

        P_W_home_B = self.teams[teamB]['home_win_prob']
        P_D_home_B = self.teams[teamB]['home_draw_prob']
        P_L_home_B = self.teams[teamB]['home_loss_prob']

        P_W_away_B = self.teams[teamB]['away_win_prob']
        P_D_away_B = self.teams[teamB]['away_draw_prob']
        P_L_away_B = self.teams[teamB]['away_loss_prob']

        # Probabilità combinata per la squadra A
        P_A_win = P_W_home_A
        P_A_draw = P_D_home_A
        P_A_loss = P_L_home_A

        P_B_win = P_W_away_B
        P_B_draw = P_D_away_B
        P_B_loss = P_L_away_B

        A_win_B = P_A_win * P_B_loss
        A_draw_B = P_A_draw * P_B_draw
        A_loss_B =  P_A_loss * P_B_win
        tot = A_win_B + A_draw_B + A_loss_B

        return {
             teamA: {
            "win": P_A_win,
            "draw": P_A_draw,
            "loss": P_A_loss
        },
        teamB: {
            "win": P_B_win,
            "draw": P_B_draw,
            "loss": P_B_loss
        },

                "AwinB": A_win_B/tot,
                "AdrawB": A_draw_B/tot,
                "AlossB": A_loss_B/tot

        }

def filter_csv(file_name, feature_list, start_dir='.'):
    for root, dirs, files in os.walk(start_dir):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            print(f"File trovato: {file_path}")

            dataframe = pd.read_csv(file_path)
            filtered_dataframe = dataframe[feature_list]

            return filtered_dataframe




def get_first_encounters(df):
    """
    Returns a DataFrame with only the first encounter between each pair of teams.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing the matches.

    Returns:
    pd.DataFrame: A new DataFrame with only the first encounter between each pair of teams.
    """
    # Ensure the Date column is of datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Create a new column that contains the pairs of teams
    df['match_pair'] = df.apply(lambda row: tuple(sorted([row['HomeTeam'], row['AwayTeam']])), axis=1)

    # Remove duplicates based on the 'match_pair' column, keeping only the first encounter
    first_encounters = df.drop_duplicates(subset='match_pair', keep='first')

    # Remove the 'match_pair' column as it is no longer needed
    first_encounters = first_encounters.drop(columns=['match_pair'])

    return first_encounters




def get_second_encounters(df):
    """
    Returns a DataFrame containing only the second match between each pair of teams.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing the matches.

    Returns:
    pd.DataFrame: A new DataFrame with only the second match between each pair of teams.
    """
    # Ensure the 'Date' column is of datetime type
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Create a new column that contains the pairs of teams
    df['match_pair'] = df.apply(lambda row: tuple(sorted([row['HomeTeam'], row['AwayTeam']])), axis=1)

    # Identify the second encounters by keeping only the subsequent duplicates
    second_encounters = df[df.duplicated(subset='match_pair', keep='first')]

    # Drop the 'match_pair' column as it is no longer needed
    second_encounters = second_encounters.drop(columns=['match_pair'])

    return second_encounters





def process_results(df):
    """
    Process the match results and return a JSON structure with statistics for each team.

    Parameters:
    df (pd.DataFrame): The DataFrame containing match data.

    Returns:
    dict: A dictionary with statistics for each team.
    """
    # Initialize the result dictionary
    team_stats = {}

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row['result_1X2']

        # Ensure both teams are in the dictionary
        if home_team not in team_stats:
            team_stats[home_team] = {'home': {'tot_match': 0, 'wins': 0, 'draws': 0, 'losses': 0},
                                     'away': {'tot_match': 0, 'wins': 0, 'draws': 0, 'losses': 0}}
        if away_team not in team_stats:
            team_stats[away_team] = {'home': {'tot_match': 0, 'wins': 0, 'draws': 0, 'losses': 0},
                                     'away': {'tot_match': 0, 'wins': 0, 'draws': 0, 'losses': 0}}

        # Update home team stats
        team_stats[home_team]['home']['tot_match'] += 1
        if result == '1':
            team_stats[home_team]['home']['wins'] += 1
            team_stats[away_team]['away']['losses'] += 1
        elif result == 'X':
            team_stats[home_team]['home']['draws'] += 1
            team_stats[away_team]['away']['draws'] += 1
        elif result == '2':
            team_stats[home_team]['home']['losses'] += 1
            team_stats[away_team]['away']['wins'] += 1

        # Update away team stats
        team_stats[away_team]['away']['tot_match'] += 1

    return team_stats

def get_match_result(df, home_team, away_team):
    """
    Returns the result of the match between the specified home and away teams.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the matches.
    home_team (str): The name of the home team.
    away_team (str): The name of the away team.

    Returns:
    pd.Series: A Series containing the match details if found, otherwise None.
    """
    match = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]
    if not match.empty:
        return match.iloc[0]  # Return the first matching result
    else:
        return None
def bayesian_process(df):
    df = get_first_encounters(df)
    # Process the results to get the team statistics
    team_stats = process_results(df)
    # Convert the dictionary to a JSON string for display or saving
    team_stats_json = json.dumps(team_stats, indent=4)
    return TeamProbability(team_stats_json)

def main():
    #variable
    file_name = "serie_a_npm=5.csv"
    season = 0

    #ONLY A SEASON
    feature_list = [features[2], features[3], features[4], features[5], features[6], features[7], features[8],
                    features[9]]
    df = filter_csv(file_name, feature_list, start_dir='../tests')
    df =  df[df['season'] == season]

    #Bayesian learning
    df_first = df.copy()
    team_prob = bayesian_process(df_first)

    #Emulation of subsequent matches
    df_second = df.copy()
    df_second = get_second_encounters(df_second)
    for index, row in df_second.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        probabilita = team_prob.conditional_probabilities(home_team, away_team)
        print(probabilita)

        match_result = get_match_result(df_second, home_team, away_team)
        print(match_result)
        df_first = pd.concat([df_first, pd.DataFrame([row])], ignore_index=True)
        team_prob = bayesian_process(df_first)

if __name__ == "__main__":
    main()