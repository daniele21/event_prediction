import pandas as pd

from salvo_models.src.feature_engineering import expand_features
from salvo_models.src.predict_match_outcomes import  *
from salvo_models.src.betting_tools import *
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

columns_to_drop = ['HomeTeam', 'AwayTeam', 'match_day', 'result_1X2', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points','match_outcome']
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

def train_models(train,validation,test):
    # Carica il dataset
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')

    bankroll = 1000
    #df_expanded = expand_features(df,columns_to_drop)
    df_expanded = df
    print("MODELLO CASA SCOMMESSE")

    evaluate_opponent_model(df_expanded[df_expanded['season'].isin(test)])

    print("MODELLO NOSTRO")
    o, prob, result = train_and_evaluate_model_with_crossval(df_expanded, train, validation, test)
    mapping = {'1': 0, 'X': 1, '2': 2}
    calibration_stats = evaluate_calibration(prob, df_expanded[df_expanded['season'].isin(test)]['result_1X2'].map(mapping))
    print_calibration_results(calibration_stats)
    total_bet = 0
    for i, (odds, predicted_probabilities, y_test) in enumerate(zip(o, prob, result)):

        # 1. Identifica scommesse di valore
        value_bets = find_value_bets(predicted_probabilities, odds)

        # 2. Calcola la frazione da scommettere (Criterio di Kelly)
        fraction_to_bet = []
        for j, ev in value_bets:
            fraction_to_bet.append(kelly_criterion(predicted_probabilities[j], odds[j]))
        # 3. Determina l'ammontare da scommettere
        stakes = manage_bankroll(np.array(fraction_to_bet), bankroll)

        total_cost = 0
        total_gain = 0

        # Esegui le scommesse
        total_bet, total_cost, total_gain = bet_on_best_value(stakes, total_bet, total_cost, y_test, total_gain, odds)
        # total_bet, total_cost, total_gain = bet_on_all_value(stakes, total_bet, total_cost, y_test,total_gain, odds)
        # total_bet, total_cost, total_gain = bet_on_top_two_values(value_bets, stakes, total_bet, total_cost, y_test, total_gain, odds)

        print("Spesa: ", total_cost)
        print("Guadagno: ", total_gain)
        print("Profitto: ", total_gain / total_cost)
        # print(f"Scommessa sull'evento {i}: {stake} euro")

def train_models_(train,validation,test):
    # Carica il dataset
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')

    bankroll = 1000
    #df_expanded = expand_features(df,columns_to_drop)
    df_expanded = df
    print("MODELLO CASA SCOMMESSE")

    evaluate_opponent_model(df_expanded[df_expanded['season'].isin(test)])

    print("MODELLO NOSTRO")
    o, prob, result = train_and_evaluate_model_with_crossval(df_expanded, train, validation, test)
    mapping = {'1': 0, 'X': 1, '2': 2}
    calibration_stats = evaluate_calibration(prob, df_expanded[df_expanded['season'].isin(test)]['result_1X2'].map(mapping))
    print_calibration_results(calibration_stats)
    total_bet = 0
    for i, (odds, predicted_probabilities, y_test) in enumerate(zip(o, prob, result)):

        # 1. Identifica scommesse di valore
        value_bets = find_value_bets(predicted_probabilities, odds)

        # 2. Calcola la frazione da scommettere (Criterio di Kelly)
        fraction_to_bet = []
        for j, ev in value_bets:
            fraction_to_bet.append(kelly_criterion(predicted_probabilities[j], odds[j]))
        # 3. Determina l'ammontare da scommettere
        stakes = manage_bankroll(np.array(fraction_to_bet), bankroll)

        total_cost = 0
        total_gain = 0

        # Esegui le scommesse
        total_bet, total_cost, total_gain = bet_on_best_value(stakes, total_bet, total_cost, y_test, total_gain, odds)
        # total_bet, total_cost, total_gain = bet_on_all_value(stakes, total_bet, total_cost, y_test,total_gain, odds)
        # total_bet, total_cost, total_gain = bet_on_top_two_values(value_bets, stakes, total_bet, total_cost, y_test, total_gain, odds)

        print("Spesa: ", total_cost)
        print("Guadagno: ", total_gain)
        print("Profitto: ", total_gain / total_cost)
        # print(f"Scommessa sull'evento {i}: {stake} euro")

def bet_on_best_value(stakes,total_bet,total_cost,y_test,total_gain,odds):
    bet = max(stakes)
    if bet > 0:
        bet_index = np.argmax(stakes)
        total_bet += 1
        total_cost += int(bet)
        if bet_index == y_test:
            total_gain += int(bet) * odds[bet_index]
    return total_bet, total_cost, total_gain


def bet_on_all_value(stakes,total_bet,total_cost,y_test,total_gain,odds):

    for j, bet in enumerate(stakes):
        if bet > 0:
            total_bet += 1
            total_cost += int(bet)
            if j == y_test:
                total_gain += int(bet) * odds[j]
    return total_bet, total_cost, total_gain

def bet_on_top_two_values(value_bets, stakes, total_bet, total_cost, y_test, total_gain, odds):
    for i, stake in zip([i for i, _ in value_bets], stakes):
        # Trova i due migliori valori di stake
        sorted_stakes = sorted(enumerate(stake), key=lambda x: x[1], reverse=True)
        top_two = sorted_stakes[:2]  # Prendi i due migliori
        bet_index1, bet1 = top_two[0]
        bet_index2, bet2 = top_two[1]

        # Se entrambi i valori sono positivi, procedi con la scommessa doppia
        if bet1 > 0 and bet2 > 0:
            # Aggiorna le statistiche per la prima scommessa
            total_bet += 1
            total_cost += int(bet1+bet2)
            if bet_index1 == y_test.iloc[i] or bet_index2 == y_test.iloc[i] :
                total_gain += int(bet1+bet2) * 1/(1/odds[i][bet_index1]+ 1/odds[i][bet_index2])

        elif bet1 > 0:
            total_bet += 1
            total_cost += int(bet1)
            if bet_index1 == y_test.iloc[i]:
                total_gain += int(bet1) * odds[i][bet_index1]

    return total_bet, total_cost, total_gain


def using_models_only_on_split(test, expected_value,kelly_perentage_request):
    # Carica il dataset
    medie = []
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')

    bankroll_new = 1000
    df_expanded = expand_features(df,columns_to_drop)
    df_expanded = df_expanded.dropna()
    tot_campionato= 0
    gain_campionato= 0
    profitto_giornate = []
    media_profitto = []
    total_bet = 0
    more_monay = 0
    for j, day in enumerate(range(5,39)):
        bankroll = bankroll_new
        day_extraction = df_expanded[df_expanded['match_day'] == day]
        match_extraction = day_extraction[day_extraction['season'].isin(test)]

        results = load_and_predict_all_models(day_extraction, test)
        cost_gioranta = 0
        gain_giornata = 0
        o, prob, result = results[0]
        total_day_match_on = 0
        for i, (odds, predicted_probabilities, y_test) in enumerate(zip(o, prob, result)):


            # 1. Identifica scommesse di valore
            value_bets = find_value_bets(predicted_probabilities, odds)

            # 2. Calcola la frazione da scommettere (Criterio di Kelly)
            check = False
            for j, ev in value_bets:
                if ev < expected_value:
                    continue
                kelly_perentage = kelly_criterion(predicted_probabilities[j], odds[j])
                if kelly_perentage < kelly_perentage_request:
                    continue
                if check:
                    continue
                check = True
                total_day_match_on += 1

        for i, (odds, predicted_probabilities, y_test) in enumerate(zip(o, prob, result)):
            # 1. Identifica scommesse di valore
            value_bets = find_value_bets(predicted_probabilities, odds)

            fraction_to_bet = []
            for j, ev in value_bets:
                if ev < expected_value:
                    fraction_to_bet.append(kelly_criterion(0, odds[j]))
                    continue
                kelly_perentage = kelly_criterion(predicted_probabilities[j], odds[j])
                if kelly_perentage < kelly_perentage_request:
                    fraction_to_bet.append(0)
                    continue
                fraction_to_bet.append(kelly_perentage)


            slit_banckroll = np.full(total_day_match_on, bankroll / total_day_match_on)  # creo un array che splitta il banckroll a tutte le squadre
            # 3. Determina l'ammontare da scommettere
            stakes = manage_bankroll(np.array(fraction_to_bet), slit_banckroll[0])

            file_name = f'.\\salvo_models\\transaction_records\\forecasts{test[0]}.csv'
            # Crea un dizionario con i dati della partita
            data = {
                'HomeTeam': match_extraction.iloc[i]['HomeTeam'],
                'AwayTeam': match_extraction.iloc[i]['AwayTeam'],
                'Bookmaker_prob1': 1 / match_extraction.iloc[i]['bet_1'],
                'Bookmaker_probX': 1 / match_extraction.iloc[i]['bet_X'],
                'Bookmaker_prob2': 1 / match_extraction.iloc[i]['bet_2'],
                'My_prob1': predicted_probabilities[0],
                'My_probX': predicted_probabilities[1],
                'My_prob2': predicted_probabilities[2],
                'Expected_value1': value_bets[0][1],
                'Expected_valueX': value_bets[1][1],
                'Expected_value2': value_bets[2][1],
                'Kelly1': fraction_to_bet[0],
                'KellyX': fraction_to_bet[1],
                'Kelly2': fraction_to_bet[2],
                'Bankroll': bankroll,
                'Bet1': stakes[0],
                'BetX': stakes[1],
                'Bet2': stakes[2]
            }
            # Crea un DataFrame con i dati
            df_to_append = pd.DataFrame([data])

            # Se il file non esiste, crealo e scrivi l'header
            if not os.path.exists(file_name):
                df_to_append.to_csv(file_name, index=False, mode='w')  # Scrivi i dati con l'header
            else:
                # Se il file esiste già, fai l'append senza scrivere l'header
                df_to_append.to_csv(file_name, index=False, mode='a', header=False)

            total_cost = 0
            total_gain = 0

            # Esegui le scommesse
            total_bet, total_cost, total_gain = bet_on_best_value(stakes,total_bet,total_cost,y_test,total_gain,odds)
            #total_bet, total_cost, total_gain = bet_on_all_value(stakes, total_bet, total_cost, y_test,total_gain, odds)
            #total_bet, total_cost, total_gain = bet_on_top_two_values(value_bets, stakes, total_bet, total_cost, y_test, total_gain, odds)
            slit_banckroll[0] = total_gain
            cost_gioranta +=total_cost
            gain_giornata += total_gain

            # print(f"Scommessa sull'evento {i}: {stake} euro")
        bankroll_new = bankroll - cost_gioranta + gain_giornata
     #   if bankroll_new < 1000:
     #       more_monay += 1000 - bankroll_new
     #       bankroll_new = 1000
        if cost_gioranta == 0:
            continue
        tot_campionato += cost_gioranta
        gain_campionato += gain_giornata
        profitto_giornate.append(gain_giornata/cost_gioranta-1)
        media_profitto.append(gain_campionato / tot_campionato - 1)


        print(f"\nSpesa giornata {day}: ",cost_gioranta)
        print(f"Guadagno giornata {day}: ", gain_giornata)
        print(f"Profitto/Perdita giornata {day} ",  (gain_giornata/cost_gioranta-1))

    print(f"\nSpesa campionato {test[0]}: ", tot_campionato)
    print(f"Guadagnamo campionato {test[0]}: ", gain_campionato)
    print(f"Profitto/Perdita campionato  {test[0]}:", gain_campionato / tot_campionato - 1)
    print(f"Scommesse totali in tutto il campionato: {total_bet}")
    print('bankroll_finale:', bankroll_new, gain_campionato-tot_campionato)
    #print('more_monay:', more_monay)

def using_models_same_split(test, expected_value,kelly_perentage_request):
    # Carica il dataset
    medie = []
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')

    bankroll_new = 1000
    df_expanded = expand_features(df,columns_to_drop)
    df_expanded = df_expanded.fillna(0)
    df_expanded = df_expanded.dropna()
    tot_campionato= 0
    gain_campionato= 0
    profitto_giornate = []
    media_profitto = []
    total_bet = 0
    more_monay = 0
    for j, day in enumerate(range(6,34)):
        bankroll = bankroll_new
        day_extraction = df_expanded[df_expanded['match_day'] == day]
        match_extraction = day_extraction[day_extraction['season'].isin(test)]
        total_day_match = len(match_extraction)
        slit_banckroll = np.full(total_day_match, bankroll/total_day_match) #creo un array che splitta il banckroll a tutte le squadre
        results = load_and_predict_all_models(day_extraction, test)
        cost_gioranta = 0
        gain_giornata = 0
        o, prob, result = results[0]
        for i, (odds, predicted_probabilities, y_test) in enumerate(zip(o, prob, result)):


            # 1. Identifica scommesse di valore
            value_bets = find_value_bets(predicted_probabilities, odds)

            # 2. Calcola la frazione da scommettere (Criterio di Kelly)
            fraction_to_bet = []
            for j, ev in value_bets:
                if ev < expected_value:
                    fraction_to_bet.append(kelly_criterion(0, odds[j]))
                    continue
                kelly_perentage = kelly_criterion(predicted_probabilities[j], odds[j])
                if kelly_perentage < kelly_perentage_request:
                    fraction_to_bet.append(0)
                    continue
                fraction_to_bet.append(kelly_perentage)
            # 3. Determina l'ammontare da scommettere
            stakes = manage_bankroll(np.array(fraction_to_bet), slit_banckroll[i])

            file_name = f'.\\salvo_models\\transaction_records\\forecasts{test[0]}.csv'
            # Crea un dizionario con i dati della partita
            data = {
                'HomeTeam': match_extraction.iloc[i]['HomeTeam'],
                'AwayTeam': match_extraction.iloc[i]['AwayTeam'],
                'Bookmaker_prob1': 1 / match_extraction.iloc[i]['bet_1'],
                'Bookmaker_probX': 1 / match_extraction.iloc[i]['bet_X'],
                'Bookmaker_prob2': 1 / match_extraction.iloc[i]['bet_2'],
                'My_prob1': predicted_probabilities[0],
                'My_probX': predicted_probabilities[1],
                'My_prob2': predicted_probabilities[2],
                'Expected_value1': value_bets[0][1],
                'Expected_valueX': value_bets[1][1],
                'Expected_value2': value_bets[2][1],
                'Kelly1': fraction_to_bet[0],
                'KellyX': fraction_to_bet[1],
                'Kelly2': fraction_to_bet[2],
                'Bankroll': bankroll,
                'Bet1': stakes[0],
                'BetX': stakes[1],
                'Bet2': stakes[2]
            }
            # Crea un DataFrame con i dati
            df_to_append = pd.DataFrame([data])

            # Se il file non esiste, crealo e scrivi l'header
            if not os.path.exists(file_name):
                df_to_append.to_csv(file_name, index=False, mode='w')  # Scrivi i dati con l'header
            else:
                # Se il file esiste già, fai l'append senza scrivere l'header
                df_to_append.to_csv(file_name, index=False, mode='a', header=False)

            total_cost = 0
            total_gain = 0

            # Esegui le scommesse
            total_bet, total_cost, total_gain = bet_on_best_value(stakes,total_bet,total_cost,y_test,total_gain,odds)
            #total_bet, total_cost, total_gain = bet_on_all_value(stakes, total_bet, total_cost, y_test,total_gain, odds)
            #total_bet, total_cost, total_gain = bet_on_top_two_values(value_bets, stakes, total_bet, total_cost, y_test, total_gain, odds)
            slit_banckroll[i] = total_gain
            cost_gioranta +=total_cost
            gain_giornata += total_gain

            # print(f"Scommessa sull'evento {i}: {stake} euro")
        bankroll_new = bankroll - cost_gioranta + gain_giornata
     #   if bankroll_new < 1000:
     #       more_monay += 1000 - bankroll_new
     #       bankroll_new = 1000
        if cost_gioranta == 0:
            continue
        tot_campionato += cost_gioranta
        gain_campionato += gain_giornata
        profitto_giornate.append(gain_giornata/cost_gioranta-1)
        media_profitto.append(gain_campionato / tot_campionato - 1)


        print(f"\nSpesa giornata {day}: ",cost_gioranta)
        print(f"Guadagno giornata {day}: ", gain_giornata)
        print(f"Profitto/Perdita giornata {day} ",  (gain_giornata/cost_gioranta-1))

    print(f"\nSpesa campionato {test[0]}: ", tot_campionato)
    print(f"Guadagnamo campionato {test[0]}: ", gain_campionato)
    print(f"Profitto/Perdita campionato  {test[0]}:", gain_campionato / tot_campionato - 1)
    print(f"Scommesse totali in tutto il campionato: {total_bet}")
    print('bankroll_finale:', bankroll_new, gain_campionato-tot_campionato)
    #print('more_monay:', more_monay)


def using_daniel_models(test, expected_value,kelly_perentage_request):
    def extract_match_info(match_day):
        # Caricare il file CSV
        df = pd.read_csv("./export_8-33_2324.csv")

        # Filtrare i dati per la giornata specifica
        filtered_df = df[df['match_day'] == match_day]

        # Estrarre le quote del bookmaker
        bookmaker_odds = filtered_df[['bet_1', 'bet_X', 'bet_2']].values

        # Estrarre le probabilità del modello
        model_probabilities = filtered_df[['1', 'X', '2']].values

        # Estrarre i risultati effettivi
        actual_results = filtered_df['result_1X2'].values

        # Formattare l'output
        return  bookmaker_odds, model_probabilities, actual_results

    # Carica il dataset
    medie = []
    df = pd.read_csv('./resources/serie_a/serie_a_npm=5_20240730_132753.csv')

    bankroll_new = 1000

    tot_campionato= 0
    gain_campionato= 0
    profitto_giornate = []
    media_profitto = []
    total_bet = 0
    more_monay = 0
    df_expanded = pd.read_csv("./export_8-33_2324.csv")
    for j, day in enumerate(range(8,33)):
        bankroll = bankroll_new
        day_extraction = df_expanded[df_expanded['match_day'] == day]
        match_extraction = day_extraction[day_extraction['season'].isin(test)]
        total_day_match = len(match_extraction)
        slit_banckroll = np.full(total_day_match, bankroll/total_day_match) #creo un array che splitta il banckroll a tutte le squadre
        results = extract_match_info(day)
        cost_gioranta = 0
        gain_giornata = 0
        o, prob, result = results
        for i, (odds, predicted_probabilities, y_test) in enumerate(zip(o, prob, result)):


            # 1. Identifica scommesse di valore
            value_bets = find_value_bets(predicted_probabilities, odds)

            # 2. Calcola la frazione da scommettere (Criterio di Kelly)
            fraction_to_bet = []
            for j, ev in value_bets:
                if ev < expected_value:
                    fraction_to_bet.append(kelly_criterion(0, odds[j]))
                    continue
                kelly_perentage = kelly_criterion(predicted_probabilities[j], odds[j])
                if kelly_perentage < kelly_perentage_request:
                    fraction_to_bet.append(0)
                    continue
                fraction_to_bet.append(kelly_perentage)
            # 3. Determina l'ammontare da scommettere
            stakes = manage_bankroll(np.array(fraction_to_bet), slit_banckroll[i])

            file_name = f'.\\salvo_models\\transaction_records\\forecasts{test[0]}.csv'
            # Crea un dizionario con i dati della partita
            data = {
                'HomeTeam': match_extraction.iloc[i]['HomeTeam'],
                'AwayTeam': match_extraction.iloc[i]['AwayTeam'],
                'Bookmaker_prob1': 1 / match_extraction.iloc[i]['bet_1'],
                'Bookmaker_probX': 1 / match_extraction.iloc[i]['bet_X'],
                'Bookmaker_prob2': 1 / match_extraction.iloc[i]['bet_2'],
                'My_prob1': predicted_probabilities[0],
                'My_probX': predicted_probabilities[1],
                'My_prob2': predicted_probabilities[2],
                'Expected_value1': value_bets[0][1],
                'Expected_valueX': value_bets[1][1],
                'Expected_value2': value_bets[2][1],
                'Kelly1': fraction_to_bet[0],
                'KellyX': fraction_to_bet[1],
                'Kelly2': fraction_to_bet[2],
                'Bankroll': bankroll,
                'Bet1': stakes[0],
                'BetX': stakes[1],
                'Bet2': stakes[2]
            }
            # Crea un DataFrame con i dati
            df_to_append = pd.DataFrame([data])

            # Se il file non esiste, crealo e scrivi l'header
            if not os.path.exists(file_name):
                df_to_append.to_csv(file_name, index=False, mode='w')  # Scrivi i dati con l'header
            else:
                # Se il file esiste già, fai l'append senza scrivere l'header
                df_to_append.to_csv(file_name, index=False, mode='a', header=False)

            total_cost = 0
            total_gain = 0

            # Esegui le scommesse
            total_bet, total_cost, total_gain = bet_on_best_value(stakes,total_bet,total_cost,y_test,total_gain,odds)
            #total_bet, total_cost, total_gain = bet_on_all_value(stakes, total_bet, total_cost, y_test,total_gain, odds)
            #total_bet, total_cost, total_gain = bet_on_top_two_values(value_bets, stakes, total_bet, total_cost, y_test, total_gain, odds)
            slit_banckroll[i] = total_gain
            cost_gioranta +=total_cost
            gain_giornata += total_gain

            # print(f"Scommessa sull'evento {i}: {stake} euro")
        bankroll_new = bankroll - cost_gioranta + gain_giornata
     #   if bankroll_new < 1000:
     #       more_monay += 1000 - bankroll_new
     #       bankroll_new = 1000
        if cost_gioranta == 0:
            continue
        tot_campionato += cost_gioranta
        gain_campionato += gain_giornata
        profitto_giornate.append(gain_giornata/cost_gioranta-1)
        media_profitto.append(gain_campionato / tot_campionato - 1)


        print(f"\nSpesa giornata {day}: ",cost_gioranta)
        print(f"Guadagno giornata {day}: ", gain_giornata)
        print(f"Profitto/Perdita giornata {day} ",  (gain_giornata/cost_gioranta-1))

    print(f"\nSpesa campionato {test[0]}: ", tot_campionato)
    print(f"Guadagnamo campionato {test[0]}: ", gain_campionato)
    print(f"Profitto/Perdita campionato  {test[0]}:", gain_campionato / tot_campionato - 1)
    print(f"Scommesse totali in tutto il campionato: {total_bet}")
    print('bankroll_finale:', bankroll_new, gain_campionato-tot_campionato)
    #print('more_monay:', more_monay)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# import the needed libraries
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    seed = 42
    expected_value = 3
    kelly_perentage_request = 0.7
    set_seed(seed)
    train = [1011,1112,1213,1314,1415,1516,1617,1718,1819,1920]
    validation = [1718]
    test = [2021]
    target = [2122]

    train_models(train,validation,test)
    #using_models_only_on_split(target,expected_value,kelly_perentage_request)
    #using_models_same_split(target,expected_value,kelly_perentage_request)
    #using_daniel_models([2324], expected_value, kelly_perentage_request)
