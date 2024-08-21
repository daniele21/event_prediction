import pandas as pd
from salvo_models.src.feature_engineering import expand_features
from salvo_models.src.predict_match_outcomes import  *
from salvo_models.src.betting_tools import *
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def evaluate_opponent_model(df_expanded):
    inv_sum = 1 / df_expanded['bet_1'] + 1 / df_expanded['bet_2'] + 1 / df_expanded['bet_X']

    col1 = (inv_sum / df_expanded['bet_1']).values
    col2 = (inv_sum / df_expanded['bet_X']).values
    col3 = (inv_sum / df_expanded['bet_2']).values

    # Creazione dell'array numpy con 3 colonne
    result_array = np.column_stack((col1, col2, col3))
    mapping = {'1': 0, 'X': 1, '2': 2}

    calibration_stats = evaluate_calibration(result_array, df_expanded['result_1X2'].map(mapping))
    print_calibration_results(calibration_stats)

def main():
    # Carica il dataset
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')
    train = [1,2,3,4,5,6,7,8]
    test = [9]
    bankroll = 1000
    df_expanded = expand_features(df)

    print("MODELLO CASA SCOMMESSE")
    evaluate_opponent_model(df_expanded[df_expanded['season'].isin(test)])

    print("MODELLO NOSTRO")
    odds, predicted_probabilities, y_test = train_and_evaluate_model(df_expanded, train, test)

    # 1. Identifica scommesse di valore
    value_bets = find_value_bets(predicted_probabilities, odds)

    # 2. Calcola la frazione da scommettere (Criterio di Kelly)
    fraction_to_bet = [kelly_criterion(predicted_probabilities[i], odds[i]) for i, ev in value_bets]

    # 3. Determina l'ammontare da scommettere
    stakes = [manage_bankroll(fraction, bankroll) for fraction in fraction_to_bet]

    total_cost = 0
    total_gain = 0
    # Esegui le scommesse
    for i, stake in zip([i for i, _ in value_bets], stakes):
        for j, bet in enumerate(stake):
            if bet > 0:
                total_cost += bet
                if j == y_test.iloc[i]:
                    total_gain += bet * odds[i][j]

    print("Spesa: ", total_cost)
    print("Guadagno: ", total_gain)
    print("Profitto: ", total_gain / total_cost)
    # print(f"Scommessa sull'evento {i}: {stake} euro")


def using_models():
    # Carica il dataset
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')
    test = [13]
    bankroll = 10
    df_expanded = expand_features(df)
    df_expanded = df_expanded.dropna()
    tot_campionato= 0
    gain_campionato= 0
    profitto_giornate = []
    media_profitto = []

    for j, day in enumerate(range(5,39)):
        results = load_and_predict_all_models(df_expanded[df_expanded['match_day']==day], test)
        cost_gioranta = 0
        gain_giornata = 0
        for i, (odds, predicted_probabilities, y_test) in enumerate(results):

            # 1. Identifica scommesse di valore
            value_bets = find_value_bets(predicted_probabilities, odds)

            # 2. Calcola la frazione da scommettere (Criterio di Kelly)
            fraction_to_bet = [kelly_criterion(predicted_probabilities[i], odds[i]) for i, ev in value_bets]

            # 3. Determina l'ammontare da scommettere
            stakes = [manage_bankroll(fraction, bankroll) for fraction in fraction_to_bet]

            total_cost = 0
            total_gain = 0
            total_bet = 0
            # Esegui le scommesse
            for i, stake in zip([i for i, _ in value_bets], stakes):
                for j, bet in enumerate(stake):
                    if bet > 0:
                        total_bet += 1
                        total_cost += int(bet)
                        if j == y_test.iloc[i]:
                            total_gain += int(bet) * odds[i][j]

            cost_gioranta +=total_cost
            gain_giornata += total_gain

            # print(f"Scommessa sull'evento {i}: {stake} euro")

        tot_campionato += cost_gioranta
        gain_campionato += gain_giornata
        profitto_giornate.append(gain_giornata/cost_gioranta-1)
        media_profitto.append(gain_campionato / tot_campionato - 1)


        print("spendiamo. ",cost_gioranta)
        print("guadagnamo. ", gain_giornata)
        print("profittiamoooo. ",  gain_giornata/cost_gioranta-1)
        print(total_bet)
    cumulative_average = np.cumsum(profitto_giornate) / np.arange(1, len(profitto_giornate) + 1)

    # Genera il grafico a linee
    plt.figure(figsize=(10, 6))
    plt.plot(profitto_giornate, marker='o', linestyle='-', color='b', label='Profitto/Perdita per Giornata')
    plt.plot(media_profitto, linestyle='--', color='orange', label='Media Cumulativa Campionato')
    plt.title('Andamento del Profitto/Perdita per Giornata Calcistica e Media Cumulativa')
    plt.xlabel('Giornata Calcistica')
    plt.ylabel('Profitto/Perdita (%)')
    plt.axhline(0, color='red', linestyle='--', label='Break Even')
    plt.legend()
    plt.grid(True)

    # Salva il grafico come file PNG
    plt.savefig('profitto_giornate_con_media_campionato.png')
    print("spendiamo campionato. ", tot_campionato)
    print("guadagnamo campionato. ", gain_campionato)
    print("profitto campionato. ", gain_campionato / tot_campionato - 1)
if __name__ == "__main__":
    #main()
    using_models()
