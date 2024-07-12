import pandas as pd
import os
import json
from lib.screapig_csv import features, filter_csv,get_first_encounters,get_second_encounters,process_results,csv_validate_models_exact,csv_validate_models_double, calculate_accuracy,calculate_gains_losses


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
            home_losses = stats['home']['losses']
            home_tot = stats['home']['tot_match']
            away_wins = stats['away']['wins']
            away_draws = stats['away']['draws']
            away_losses = stats['away']['losses']
            away_tot = stats['away']['tot_match']

            stats['home_win_prob'] = home_wins / home_tot if home_tot > 0 else 0
            stats['home_draw_prob'] = home_draws / home_tot if home_tot > 0 else 0
            stats['home_loss_prob'] = home_losses / home_tot if home_tot > 0 else 0

            stats['away_win_prob'] = away_wins / away_tot if away_tot > 0 else 0
            stats['away_draw_prob'] = away_draws / away_tot if away_tot > 0 else 0
            stats['away_loss_prob'] = away_losses / away_tot if away_tot > 0 else 0

    def combinatorial_probabilities(self, teamA, teamB):
        self.calculate_probabilities()

        if teamA not in self.teams or teamB not in self.teams:
            raise ValueError("One or both teams are not in the dataset")

        # Probabilità di vittoria, pareggio e sconfitta per la squadra A e la squadra B
        P_A = self.teams[teamA]
        P_B = self.teams[teamB]

        P_A_win, P_A_draw, P_A_loss = P_A['home_win_prob'], P_A['home_draw_prob'], P_A['home_loss_prob']
        P_B_win, P_B_draw, P_B_loss = P_B['away_win_prob'], P_B['away_draw_prob'], P_B['away_loss_prob']

        # Probabilità combinata
        A_win_B = P_A_win * P_B_loss
        A_draw_B = P_A_draw * P_B_draw
        A_loss_B = P_A_loss * P_B_win
        tot = A_win_B + A_draw_B + A_loss_B

        return{
            "AwinB": A_win_B / tot,
            "AdrawB": A_draw_B / tot,
            "AlossB": A_loss_B / tot
        }


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

def combinatorial_process(df):
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
                    features[9], features[10], features[11], features[12]]
    df = filter_csv(file_name, feature_list, start_dir='../tests')
    df =  df[df['season'] == season]

    #Combinatorial learning
    df_first = df.copy()
    team_prob = combinatorial_process(df_first)

    #Emulation of subsequent matches
    df_second = df.copy()
    df_second = get_second_encounters(df_second)
    for index, row in df_second.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        quote_home = row[features[10]]
        quote_draw = row[features[11]]
        quote_way = row[features[12]]
        probability = team_prob.combinatorial_probabilities(home_team, away_team)

        match_result = get_match_result(df_second, home_team, away_team)
        df_first = pd.concat([df_first, pd.DataFrame([row])], ignore_index=True)
        team_prob = combinatorial_process(df_first)
        csv_validate_models_exact('combinatorial_probabilities_exact.csv', home_team, away_team, probability["AwinB"], probability["AdrawB"], probability["AlossB"], match_result['result_1X2'],  quote_home, quote_draw, quote_way)
        csv_validate_models_double('combinatorial_probabilities_double.csv', home_team, away_team, probability["AwinB"], probability["AdrawB"], probability["AlossB"], match_result['result_1X2'],quote_home, quote_draw, quote_way)





if __name__ == "__main__":
    main()
    print(f"combinatorial_probabilities_exact: {calculate_accuracy('combinatorial_probabilities_exact.csv')}")
    print(f"combinatorial_probabilities_double: {calculate_accuracy('combinatorial_probabilities_double.csv')}")
    print("Total gain with exact results")
    calculate_gains_losses("combinatorial_probabilities_exact.csv", 2)
    print("Total gain with double results")
    calculate_gains_losses("combinatorial_probabilities_double.csv", 2)