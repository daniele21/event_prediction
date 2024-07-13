import pandas as pd
import os
import json
import numpy as np
from lib.screapig_csv import features, filter_csv, get_first_encounters, get_second_encounters, process_results, csv_validate_models_exact, csv_validate_models_double, calculate_accuracy, calculate_gains_losses, kelly_calculate_gains_losses
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import factorial  # Importa la funzione factorial da scipy.special
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
    season = 1

    feature_list = features
    df = filter_csv(file_name, feature_list, start_dir='../tests')
    df = df[df['season'] == season]
    df.rename(columns=lambda x: x.replace('-', '_').replace(' ', '_'), inplace=True)
    df_first = df.copy()
    df_first = get_first_encounters(df_first)
    # Fit Poisson regression model for home and away goals
    data_cleaned = df_first.dropna()

    formula_home = 'home_goals ~ HomeTeam + AwayTeam + HOME_last_1_away + HOME_last_1 + HOME_last_1_home + HOME_last_2 + HOME_last_2_away + HOME_last_3 + HOME_last_2_home + HOME_last_4 + HOME_last_3_away + HOME_last_5 + HOME_last_3_home + HOME_last_4_away + HOME_last_4_home + HOME_last_5_away + HOME_last_5_home'
    formula_away = 'away_goals ~ HomeTeam + AwayTeam + AWAY_last_1_home + AWAY_last_1 + AWAY_last_1_away + AWAY_last_2 + AWAY_last_2_home + AWAY_last_3 + AWAY_last_2_away + AWAY_last_4 + AWAY_last_3_home + AWAY_last_5 + AWAY_last_3_away + AWAY_last_4_home + AWAY_last_4_away + AWAY_last_5_home + AWAY_last_5_away'
    #poisson_model_home = smf.poisson(formula_home, data=data_cleaned).fit()
    #poisson_model_away = smf.poisson(formula_away, data=data_cleaned).fit()
    poisson_model_home = smf.poisson('home_goals ~ HomeTeam + AwayTeam + HOME_last_5 +HOME_last_5_home ', data=data_cleaned).fit()
    poisson_model_away = smf.poisson('away_goals ~ HomeTeam + AwayTeam + AWAY_last_5 +AWAY_last_5_away', data=data_cleaned).fit()

    df_second = df.copy()
    df_second = get_second_encounters(df_second)

    for index, row in df_second.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        quote_home = row[features[10]]
        quote_draw = row[features[11]]
        quote_way = row[features[12]]

        home_lambda, away_lambda = calculate_expected_goals(poisson_model_home, poisson_model_away, home_team, away_team)
        home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_lambda, away_lambda)
        csv_validate_models_exact('poisson_exact.csv', home_team, away_team, home_win_prob,
                                  draw_prob, away_win_prob, row['result_1X2'], quote_home,
                                  quote_draw, quote_way)
        csv_validate_models_double('poisson_double.csv', home_team, away_team, home_win_prob,
                                  draw_prob, away_win_prob, row['result_1X2'], quote_home,
                                  quote_draw, quote_way)


def best_model():
    best_min_prob = None
    best_min_gain = None
    max_gain = float('-inf')

    for min_prob in np.arange(0.5, 1.0, 0.05):
        for min_gain in np.arange(1, 10, 1):
            gain_exact = calculate_gains_losses("poisson_exact.csv", value, min_prob, min_gain)
            gain_double = calculate_gains_losses("poisson_double.csv", value, min_prob, min_gain)
            gain_kelly_exact = kelly_calculate_gains_losses("poisson_exact.csv", value, min_prob, min_gain)
            gain_kelly_double = kelly_calculate_gains_losses("poisson_double.csv", value, min_prob, min_gain)

            total_gain = gain_exact + gain_double + gain_kelly_exact + gain_kelly_double

            if total_gain > max_gain:
                max_gain = total_gain
                best_min_prob = min_prob
                best_min_gain = min_gain

    print(f"Best min_prob: {best_min_prob}")
    print(f"Best min_gain: {best_min_gain}")
    print(f"Max gain: {max_gain}")
    return best_min_prob, best_min_gain

if __name__ == "__main__":
    main()
    print(f"poisson_exact: {calculate_accuracy('poisson_exact.csv')}")
    print(f"poisson_double: {calculate_accuracy('poisson_double.csv')}")
    print("Total gain with exact results")
    value = 100
    #min_prob, min_gain = best_model()
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
