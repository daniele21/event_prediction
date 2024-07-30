
from lib.screapig_csv import features, filter_csv, get_first_encounters, get_second_encounters, process_results, csv_validate_models_exact, csv_validate_models_double, calculate_accuracy, calculate_gains_losses, kelly_calculate_gains_losses,best_model
from scipy.special import factorial  # Importa la funzione factorial da scipy.special
import statsmodels.formula.api as smf
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg')  # Usa questo se vuoi visualizzare i grafici in una finestra grafica
import pandas as pd

def optimize_probabilities(data):
    """
    Ottimizza i pesi delle probabilità di vittoria, pareggio e sconfitta
    utilizzando i dati forniti nel dataframe.

    Args:
    data (pd.DataFrame): DataFrame contenente le colonne 'A_win_prob', 'A_draw_prob',
                         'A_loss_prob' e 'Ground_Truth'.

    Returns:
    np.ndarray: Pesi ottimizzati per le probabilità di vittoria, pareggio e sconfitta.
    """
    # Preparazione dei dati
    X = data[['A_win_prob', 'A_draw_prob', 'A_loss_prob']].values

    # Creazione del ground truth
    y_win = np.array([0 if gt == 'X' else 1 if gt == '1' else 0 for gt in data['Ground_Truth']])
    y_draw = np.array([1 if gt == 'X' else 0 for gt in data['Ground_Truth']])
    y_loss = np.array([0 if gt == 'X' else 0 if gt == '1' else 1 for gt in data['Ground_Truth']])

    # Funzione obiettivo complessiva con pesi
    def overall_objective(weights, X, y_win, y_draw, y_loss):
        win_prob = np.dot(X, weights[:3])
        draw_prob = np.dot(X, weights[3:6])
        loss_prob = np.dot(X, weights[6:])

        total_error = np.mean((win_prob - y_win) ** 2 + (draw_prob - y_draw) ** 2 + (loss_prob - y_loss) ** 2)
        return total_error

    # Limiti per i pesi (possono essere negativi se necessario)
    bounds = [(-1, 1)] * 9

    # Inizializzazione dei pesi
    initial_weights = np.ones(9) / 9

    # Ottimizzazione
    solution = minimize(overall_objective, initial_weights, args=(X, y_win, y_draw, y_loss), bounds=bounds)

    # Pesos ottimizzati
    optimized_weights = solution.x

    # Calcolo delle probabilità combinate
    combined_win_prob = np.dot(X, optimized_weights[:3])
    combined_draw_prob = np.dot(X, optimized_weights[3:6])
    combined_loss_prob = np.dot(X, optimized_weights[6:])

    # Sostituisci le colonne con le nuove probabilità combinate
    data['A_win_prob'] = combined_win_prob
    data['A_draw_prob'] = combined_draw_prob
    data['A_loss_prob'] = combined_loss_prob

    # Valutazione del modello
    mse_win = mean_squared_error(y_win, combined_win_prob)
    mse_draw = mean_squared_error(y_draw, combined_draw_prob)
    mse_loss = mean_squared_error(y_loss, combined_loss_prob)

    print(f"Valore ottimizzato dei pesi: {optimized_weights}")
    print(f"MSE Win: {mse_win}")
    print(f"MSE Draw: {mse_draw}")
    print(f"MSE Loss: {mse_loss}")

    return optimized_weights

def apply_weights(home_win_prob, draw_prob, away_win_prob, weights):
    X = np.array([home_win_prob, draw_prob, away_win_prob])
    home_win_prob_new = np.dot(X, weights[:3])
    draw_prob_new = np.dot(X, weights[3:6])
    away_win_prob_new = np.dot(X, weights[6:])
    return home_win_prob_new, draw_prob_new, away_win_prob_new


# Funzione per calcolare i goal attesi
def calculate_expected_goals(model_home, model_away, home_team, away_team):
    home_goals_lambda = np.exp(model_home.params['Intercept'] +
                               model_home.params.get(f'HomeTeam[T.{home_team}]', 0) +
                               model_home.params.get(f'AwayTeam[T.{away_team}]', 0))

    away_goals_lambda = np.exp(model_away.params['Intercept'] +
                               model_away.params.get(f'HomeTeam[T.{home_team}]', 0) +
                               model_away.params.get(f'AwayTeam[T.{away_team}]', 0))

    return home_goals_lambda, away_goals_lambda


# Calcola le probabilità di ogni numero di goal
def poisson_prob(lambda_, k):
    return (lambda_ ** k) * np.exp(-lambda_) / factorial(k)


# Calcola le probabilità di vincita, pareggio e sconfitta
def calculate_match_outcome_probabilities(home_lambda, away_lambda, max_goals=10):
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0

    for i in range(max_goals):
        for j in range(max_goals):
            prob = poisson_prob(home_lambda, i) * poisson_prob(away_lambda, j)
            if i > j:
                home_win_prob += prob
            elif i == j:
                draw_prob += prob
            else:
                away_win_prob += prob

    return home_win_prob, draw_prob, away_win_prob




def main():
    # Variable
    file_name = "serie_a_npm=5.csv"
    feature_list = features
    df = filter_csv(file_name, feature_list, start_dir='../tests')
    season = 5
    df = df[df['season'] == season]
    df_train = get_first_encounters(df).copy()
    df_test = get_second_encounters(df).copy()
    #TRAIN
    df_train.rename(columns=lambda x: x.replace('-', '_').replace(' ', '_'), inplace=True)
    df_train = df_train.dropna()


    #OPTIMIZATION




    formula_home = 'home_goals ~ HomeTeam + AwayTeam + HOME_last_1_away + HOME_last_1 + HOME_last_1_home + HOME_last_2 + HOME_last_2_away + HOME_last_3 + HOME_last_2_home + HOME_last_4 + HOME_last_3_away + HOME_last_5 + HOME_last_3_home + HOME_last_4_away + HOME_last_4_home + HOME_last_5_away + HOME_last_5_home'
    formula_away = 'away_goals ~ HomeTeam + AwayTeam + AWAY_last_1_home + AWAY_last_1 + AWAY_last_1_away + AWAY_last_2 + AWAY_last_2_home + AWAY_last_3 + AWAY_last_2_away + AWAY_last_4 + AWAY_last_3_home + AWAY_last_5 + AWAY_last_3_away + AWAY_last_4_home + AWAY_last_4_away + AWAY_last_5_home + AWAY_last_5_away'
    home_features = ['HomeTeam', 'AwayTeam', 'HOME_last_1', 'HOME_last_1_home', 'HOME_last_1_away',
                     'HOME_last_2', 'HOME_last_2_home', 'HOME_last_2_away',
                     'HOME_last_3', 'HOME_last_3_home', 'HOME_last_3_away',
                     'HOME_last_4', 'HOME_last_4_home', 'HOME_last_4_away',
                     'HOME_last_5', 'HOME_last_5_home', 'HOME_last_5_away']

    away_features = ['HomeTeam', 'AwayTeam', 'AWAY_last_1', 'AWAY_last_1_home', 'AWAY_last_1_away',
                     'AWAY_last_2', 'AWAY_last_2_home', 'AWAY_last_2_away',
                     'AWAY_last_3', 'AWAY_last_3_home', 'AWAY_last_3_away',
                     'AWAY_last_4', 'AWAY_last_4_home', 'AWAY_last_4_away',
                     'AWAY_last_5', 'AWAY_last_5_home', 'AWAY_last_5_away']



    for index, row in df_test.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        quote_home = row[features[10]]
        quote_draw = row[features[11]]
        quote_way = row[features[12]]

        poisson_model_home = smf.poisson('home_goals ~ HomeTeam + AwayTeam + HOME_last_5 + HOME_last_5_home',
                                         data=df_train).fit()
        poisson_model_away = smf.poisson('away_goals ~ HomeTeam + AwayTeam + AWAY_last_5 + AWAY_last_5_away',
                                         data=df_train).fit()

        home_lambda, away_lambda = calculate_expected_goals(poisson_model_home, poisson_model_away, home_team,
                                                            away_team)

        home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_lambda, away_lambda)

        new_row = pd.DataFrame([row])

        # Concatenare il nuovo DataFrame a df_first
        df_train = pd.concat([df_train, new_row], ignore_index=True)

        csv_validate_models_exact('poisson_exact.csv', home_team, away_team, home_win_prob,
                                  draw_prob, away_win_prob, row['result_1X2'], quote_home,
                                  quote_draw, quote_way)
        csv_validate_models_double('poisson_double.csv', home_team, away_team, home_win_prob,
                                   draw_prob, away_win_prob, row['result_1X2'], quote_home,
                                   quote_draw, quote_way)







if __name__ == "__main__":
    main()
    print(f"poisson_exact: {calculate_accuracy('poisson_exact.csv')}")
    print(f"poisson_double: {calculate_accuracy('poisson_double.csv')}")
    print("Total gain with exact results")
    value = 10
    #min_prob, min_gain = best_model(value,"poisson_exact.csv","poisson_double.csv")
    min_prob, min_gain = 0.85 , 2

    gain= calculate_gains_losses("poisson_exact.csv", value, min_prob,min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('poisson_exact.csv')}")

    print("\nTotal gain with doule results")
    gain =calculate_gains_losses("poisson_double.csv", value, min_prob,min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('poisson_double.csv')}")

    print("\nTotal gain with exact results with kelly")
    gain= kelly_calculate_gains_losses("poisson_exact.csv", value, min_prob,min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('poisson_exact.csv')}")

    print("\nTotal gain with double results with kelly")
    gain= kelly_calculate_gains_losses("poisson_double.csv", value, min_prob,min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('poisson_double.csv')}")
