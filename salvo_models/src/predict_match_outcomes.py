import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPRegressor
def cleaning_and_target(df):
    df = df.copy()
    df = df.dropna()

    columns_to_drop = ['HomeTeam', 'AwayTeam', 'match_day', 'result_1X2', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points']

    df['match_outcome'] = np.where(df['home_goals'] > df['away_goals'], 0,
                                   np.where(df['home_goals'] < df['away_goals'], 2, 1))
    inv_sum = 1 / df['bet_1'] + 1 / df['bet_2'] + 1 / df['bet_X']

    col1 = (inv_sum / df['bet_1']).values
    col2 = (inv_sum / df['bet_X']).values
    col3 = (inv_sum / df['bet_2']).values
    mapping = {'1': 0, 'X': 1, '2': 2}
    results = df['result_1X2'].map(mapping)
    odds = df[['bet_1', 'bet_X', 'bet_2']].apply(
        lambda row: np.array([row['bet_1'], row['bet_X'], row['bet_2']]), axis=1).values
    df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df_cleaned['prob_1'] = col1
    df_cleaned['prob_2'] = col3
    df_cleaned['prob_X'] = col2
    return df_cleaned, results, odds

def evaluate_calibration(y_pred_proba, y_test, n_bins=10):
    """
    Valuta la calibrazione delle probabilità predette.

    Args:
    y_pred_proba (np.ndarray): Array delle probabilità predette, con una riga per ciascuna osservazione e una colonna per ciascuna classe.
    y_test (np.ndarray): Array delle classi effettive.
    n_bins (int): Numero di intervalli di probabilità da considerare (default: 10).

    Returns:
    dict: Dizionario che contiene le statistiche di calibrazione per ciascuna classe.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_stats = {i: defaultdict(lambda: {'count': 0, 'success': 0}) for i in range(y_pred_proba.shape[1])}

    for proba, true_class in zip(y_pred_proba, y_test):
        for class_idx in range(y_pred_proba.shape[1]):
            bin_idx = np.digitize(proba[class_idx], bins) - 1
            # Invece di confrontare un array, confrontiamo direttamente il valore scalare
            if true_class == class_idx:
                calibration_stats[class_idx][bin_idx]['success'] += 1
            calibration_stats[class_idx][bin_idx]['count'] += 1

    return calibration_stats


def print_calibration_results(calibration_stats, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    for class_idx, stats in calibration_stats.items():
        print(f"\nClasse {class_idx}:")
        for bin_idx in range(n_bins):
            lower_bound = bins[bin_idx]
            upper_bound = bins[bin_idx + 1]
            count = stats[bin_idx]['count']
            success = stats[bin_idx]['success']
            if count > 0:
                observed_freq = success / count
                print(f"Probabilità {lower_bound:.1f}-{upper_bound:.1f}: Frequenza osservata = {observed_freq:.2f} ({count} predizioni)")
            else:
                print(f"Probabilità {lower_bound:.1f}-{upper_bound:.1f}: Nessuna predizione")

def train_and_evaluate_model(df, train, test):
    df_train, _, _ = cleaning_and_target(df[df['season'].isin(train)])
    df_test, results, odds = cleaning_and_target(df[df['season'].isin(test)])

    X_train = df_train.drop(columns=['prob_1', 'prob_X', 'prob_2']).fillna(0)
    y_train = df_train[['prob_1', 'prob_X', 'prob_2']].astype(float)
    X_test = df_test.drop(columns=['prob_1', 'prob_X', 'prob_2']).fillna(0)
    y_test = df_test[['prob_1', 'prob_X', 'prob_2']].astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        max_iter=10000,
        hidden_layer_sizes=(50, 50, 50),
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Stampa delle prime predizioni per confronto
    print("First 10 predicted vs actual probabilities:")
    print(pd.DataFrame({
        'Predicted_1': y_pred[:, 0], 'Actual_1': y_test['prob_1'].values,
        'Predicted_X': y_pred[:, 1], 'Actual_X': y_test['prob_X'].values,
        'Predicted_2': y_pred[:, 2], 'Actual_2': y_test['prob_2'].values
    }).head(10))

    # Salva il modello
    save_model(model)

    return odds, y_pred, results

def save_model(model):
    """Salva il modello in un percorso specificato, numerando i file automaticamente."""
    model_dir = './salvo_models/models'
    os.makedirs(model_dir, exist_ok=True)  # Crea la directory se non esiste

    # Trova il primo numero disponibile per il nome del file
    model_files = os.listdir(model_dir)
    existing_numbers = [int(f.split('.')[0]) for f in model_files if f.split('.')[0].isdigit()]
    next_number = 0 if not existing_numbers else max(existing_numbers) + 1

    model_path = os.path.join(model_dir, f'{next_number}.joblib')

    # Salva il modello
    joblib.dump(model, model_path)
    print(f'Modello salvato come {model_path}')



def load_and_predict_all_models(df, test):
    """
    Carica tutti i modelli salvati e li utilizza per fare previsioni sul dataset di test.

    Args:
    df -- Il DataFrame originale da utilizzare per il test.
    test -- Lista degli indici del test set.

    Returns:
    results_list -- Lista di tuple (odds, y_pred, results) per ciascun modello.
    """
    model_dir = './salvo_models/models'
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.joblib')])
    df_test, results, odds = cleaning_and_target(df[df['season'].isin(test)])
    X_test = df_test.drop(columns=['prob_1', 'prob_X', 'prob_2']).fillna(0)

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    results_list = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)

        # Aggiunge i risultati alla lista
        results_list.append((odds, y_pred, results))

        # Stampa delle metriche per il modello corrente
        print(f"Risultati per il modello {model_file}:")
        mse = mean_squared_error(df_test[['prob_1', 'prob_X', 'prob_2']], y_pred)
        r2 = r2_score(df_test[['prob_1', 'prob_X', 'prob_2']], y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}\n")

    return results_list