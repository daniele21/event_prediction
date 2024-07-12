import os
import pandas as pd

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


def csv_validate_models_exact(file_name, teamA, teamB, A_win_prob, A_draw_prob, A_loss_prob, ground_truth,quote_home, quote_draw, quote_way):
    # Verifica se il file esiste
    path = f'./predictions_file/{file_name}'
    if os.path.exists(path):
        results_df = pd.read_csv(path)
    else:
        # Crea un DataFrame vuoto con le colonne appropriate
        columns = ['HomeTeam', 'AwayTeam', 'A_win_prob', 'A_draw_prob', 'A_loss_prob', 'Ground_Truth', 'Predicted', 'Correct']
        results_df = pd.DataFrame(columns=columns)

    # Determina la predizione basata sulla probabilità più alta
    probabilities = {
        '1': A_win_prob,
        'X': A_draw_prob,
        '2': A_loss_prob
    }
    predicted = max(probabilities, key=probabilities.get)

    # Determina se la predizione è corretta
    correct = (predicted == ground_truth)
    my_quotes = {
        '1': 1 / (A_win_prob) if (A_win_prob) != 0 else 0,
        'X': 1 / (A_draw_prob) if (A_draw_prob) != 0 else 0,
        '2': 1 / ( A_loss_prob) if (A_loss_prob) != 0 else 0
    }

    # Crea un DataFrame per la nuova riga di dati
    new_data = pd.DataFrame([{
        'HomeTeam': teamA,
        'AwayTeam': teamB,
        'my_1': my_quotes['1'],
        'my_X': my_quotes['X'],
        'my_2': my_quotes['2'],
        'GT_1': quote_home,
        'GT_X': quote_draw,
        'GT_2': quote_way,
        'A_win_prob': A_win_prob,
        'A_draw_prob': A_draw_prob,
        'A_loss_prob': A_loss_prob,
        'Ground_Truth': ground_truth,
        'Predicted': predicted,
        'Correct': correct
    }])

    # Escludi le entry vuote o tutte NA prima della concatenazione
    new_data = new_data.dropna(axis=1, how='all')
    results_df = results_df.dropna(axis=1, how='all')

    # Concatena la nuova riga al DataFrame esistente
    results_df = pd.concat([results_df, new_data], ignore_index=True)

    # Salva il DataFrame aggiornato nel file CSV
    results_df.to_csv(path, index=False)

    # Calcola e stampa il numero totale di predizioni corrette

def csv_validate_models_double(file_name, teamA, teamB, A_win_prob, A_draw_prob, A_loss_prob, ground_truth,quote_home, quote_draw, quote_way):
    # Verifica se il file esiste
    path = f'./predictions_file/{file_name}'
    if os.path.exists(path):
        results_df = pd.read_csv(path)
    else:
        # Crea un DataFrame vuoto con le colonne appropriate
        columns = ['HomeTeam', 'AwayTeam', 'A_win_prob', 'A_draw_prob', 'A_loss_prob', 'Ground_Truth', 'Predicted', 'Correct']
        results_df = pd.DataFrame(columns=columns)

    # Determina la predizione basata sulla probabilità più alta
    probabilities = {
        '1X': A_win_prob + A_draw_prob,
        '2X': A_loss_prob+ A_draw_prob,
        '12': A_win_prob + A_loss_prob
    }
    my_quotes = {
        '1X': 1 / (A_win_prob + A_draw_prob) if (A_win_prob + A_draw_prob) != 0 else 0,
        '2X': 1 / (A_loss_prob + A_draw_prob) if (A_loss_prob + A_draw_prob) != 0 else 0,
        '12': 1 / (A_win_prob + A_loss_prob) if (A_win_prob + A_loss_prob) != 0 else 0
    }
    GT_q_1x =1/( 1/ quote_home + 1/quote_draw)
    GT_q_2x = 1/(1/ quote_way + 1/quote_draw)
    GT_q_12= 1/(1/quote_home +1/quote_way)
    predicted = max(probabilities, key=probabilities.get)

    # Determina se la predizione è corretta
    if len(predicted) == 2:
        correct= ( predicted[0] == ground_truth or predicted[1] == ground_truth)



    # Crea un DataFrame per la nuova riga di dati
    new_data = pd.DataFrame([{
        'HomeTeam': teamA,
        'AwayTeam': teamB,
        'my_1X':my_quotes['1X'],
        'my_2X': my_quotes['2X'],
        'my_12': my_quotes['12'],
        'GT_1X': GT_q_1x,
        'GT_2X': GT_q_2x,
        'GT_12': GT_q_12,
        'A_win_prob': A_win_prob,
        'A_draw_prob': A_draw_prob,
        'A_loss_prob': A_loss_prob,
        'Ground_Truth': ground_truth,
        'Predicted': predicted,
        'Correct': correct
    }])

    # Escludi le entry vuote o tutte NA prima della concatenazione
    new_data = new_data.dropna(axis=1, how='all')
    results_df = results_df.dropna(axis=1, how='all')

    # Concatena la nuova riga al DataFrame esistente
    results_df = pd.concat([results_df, new_data], ignore_index=True)

    # Salva il DataFrame aggiornato nel file CSV
    results_df.to_csv(path, index=False)


def calculate_accuracy(file_name):
    path = f'./predictions_file/{file_name}'
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return

    results_df = pd.read_csv(path)

    total_records = len(results_df)
    total_correct = results_df['Correct'].sum()

    return total_correct/total_records


import pandas as pd


def calculate_gains_losses(file_path, amount,min_prob):
    path = f'./predictions_file/{file_path}'

    df = pd.read_csv(path)

    # Initialize variables
    gain = 0
    loss = 0
    total_spent = 0

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        prediction = row['Predicted']
        correct = row['Correct']
        A_win_prob = row['A_win_prob']
        A_draw_prob = row['A_draw_prob']
        A_loss_prob = row['A_loss_prob']

        # Find the column name based on the prediction
        gt_column = f"GT_{prediction}"


        if gt_column in row:
            odds = row[gt_column]
        else:
            continue  # Skip if the column is not found
        if prediction == '1':

           if A_win_prob < min_prob:
               continue
        if prediction == 'X':
           if A_draw_prob < min_prob:
               continue

        if prediction == '2':
           if A_loss_prob < min_prob:
               continue

        if prediction == '1X':
            one_draw = A_win_prob + A_draw_prob
            if one_draw < min_prob:
                continue

        if prediction == 'X2':
            two_draw = A_loss_prob + A_draw_prob
            if two_draw < min_prob:
                continue

        if prediction == '12':
            one_two = A_win_prob + A_loss_prob
            if one_two < min_prob:
                continue



        total_spent += amount

        if correct:
            gain += amount * odds
        else:
            loss += amount

    # Print the results
    print(f"Total gain: {gain}")
    print(f"Total loss: {loss}")
    print(f"Total spent: {total_spent}")
    print(f"loss percentage: {loss/total_spent}")
    print(f"gain percentage: {(gain- total_spent) / total_spent}")




def kelly_calculate_gains_losses(file_path, amount,min_prob):
    path = f'./predictions_file/{file_path}'

    df = pd.read_csv(path)

    # Initialize variables
    gain = 0
    loss = 0
    total_spent = 0

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        prediction = row['Predicted']
        correct = row['Correct']
        A_win_prob = row['A_win_prob']
        A_draw_prob = row['A_draw_prob']
        A_loss_prob = row['A_loss_prob']

        # Find the column name based on the prediction
        gt_column = f"GT_{prediction}"


        if gt_column in row:
            odds = row[gt_column]
        else:
            continue  # Skip if the column is not found
        b = odds - 1
        if prediction == '1':
           f = ((b * A_win_prob) - (1-A_win_prob))/b
           if A_win_prob < min_prob:
               continue
        if prediction == 'X':
           if A_draw_prob < min_prob:
               continue
           f = ((b * A_draw_prob) - (1-A_draw_prob))/b
        if prediction == '2':
           if A_loss_prob < min_prob:
               continue
           f = ((b * A_loss_prob) - (1-A_loss_prob))/b
        if prediction == '1X':
            one_draw = A_win_prob + A_draw_prob
            if one_draw < min_prob:
                continue
            f = ((b * one_draw) - (1 - one_draw)) / b
        if prediction == 'X2':
            two_draw = A_loss_prob + A_draw_prob
            if two_draw < min_prob:
                continue
            f = ((b * two_draw) - (1 - two_draw)) / b
        if prediction == '12':
            one_two = A_win_prob + A_loss_prob
            if one_two < min_prob:
                continue

            f = ((b * one_two) - (1 - one_two)) / b
        f = int(f)
        total_spent += amount*f

        if correct:
            gain += amount*f * odds
        else:
            loss += amount*f

    # Print the results
    print(f"Total gain: {gain}")
    print(f"Total loss: {loss}")
    print(f"Total spent: {total_spent}")
    print(f"loss percentage: {loss/total_spent}")
    print(f"gain percentage: {(gain -total_spent) / total_spent}")



