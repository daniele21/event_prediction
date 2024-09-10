import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
def cleaning_and_target(df):
    df = df.copy()
    df = df.fillna(0)
    #df = df.dropna()
    df = df[~df['match_day'].between(1, 5) & ~df['match_day'].between(34, 38)]

    columns_to_drop = ['HomeTeam', 'AwayTeam', 'match_day', 'result_1X2', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points','match_outcome']
    columns_to_drop1 = ['HomeTeam', 'AwayTeam', 'match_day', 'result_1X2', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points', 'match_outcome',
                       'prev_cum_home_points', 'prev_home_league_points', 'prev_cum_home_goals',
                       'prev_home_league_goals', 'prev_cum_away_points', 'prev_away_league_points',
                       'prev_cum_away_goals', 'prev_away_league_goals', 'HOME_last-1-away', 'HOME_last-1',
                       'AWAY_last-1-home', 'AWAY_last-1', 'AWAY_last-1-away', 'AWAY_last-2',
                       'HOME_last-1-home', 'HOME_last-2', 'HOME_l   ast-2-away', 'HOME_last-3',
                       'AWAY_last-2-home', 'AWAY_last-3', 'HOME_last-2-home', 'HOME_last-4',
                       'AWAY_last-2-away', 'AWAY_last-4', 'HOME_last-3-away', 'HOME_last-5',
                       'AWAY_last-3-home', 'AWAY_last-5', 'HOME_last-3-home', 'AWAY_last-3-away',
                       'HOME_last-4-away', 'AWAY_last-4-home', 'HOME_last-4-home', 'AWAY_last-4-away',
                       'HOME_last-5-away', 'AWAY_last-5-home', 'HOME_last-5-home', 'AWAY_last-5-away',
                       'point_diff_rolling_mean_3', 'point_diff_rolling_mean_5', 'point_diff_ewm_5',
                       'point_diff_diff_1', 'point_diff_rolling_std_3', 'point_diff_rolling_std_5',
                       'point_diff_cumsum', 'goals_diff_rolling_mean_3', 'goals_diff_rolling_mean_5',
                       'goals_diff_ewm_5', 'goals_diff_diff_1', 'goals_diff_diff_2']

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
    #df_cleaned['prob_1'] = col1
    #df_cleaned['prob_2'] = col3
    #df_cleaned['prob_X'] = col2
    return df_cleaned, results, odds
def cleaning_and_target_(df):
    df = df.copy()
    df = df.dropna()

    columns_to_drop = ['HomeTeam', 'AwayTeam', 'match_day', 'result_1X2', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points','match_outcome']

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

def train_and_evaluate_model_(df, train,validation, test):
    df_train, _, _ = cleaning_and_target_(df[df['season'].isin(train)])
    df_test, results, odds = cleaning_and_target_(df[df['season'].isin(test)])

    X_train = df_train.drop(columns=['prob_1', 'prob_X', 'prob_2']).fillna(0)
    y_train = df_train[['prob_1', 'prob_X', 'prob_2']].astype(float)
    X_test = df_test.drop(columns=['prob_1', 'prob_X', 'prob_2']).fillna(0)
    y_test = df_test[['prob_1', 'prob_X', 'prob_2']].astype(float)

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train
    X_test_scaled = X_test
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Stampa delle prime predizioni per confronto
    #print("First 10 predicted vs actual probabilities:")
    #print(pd.DataFrame({
    #    'Predicted_1': y_pred[:, 0], 'Actual_1': y_test['prob_1'].values,
    #    'Predicted_X': y_pred[:, 1], 'Actual_X': y_test['prob_X'].values,
    #    'Predicted_2': y_pred[:, 2], 'Actual_2': y_test['prob_2'].values
    #}).head(10))

    # Salva il modello
    save_model(model)

    return odds, y_pred, results


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import numpy as np
import random
def set_seed(seed=42):
    random.seed(seed)  # Imposta il seed per il modulo random di Python
    np.random.seed(seed)  # Imposta il seed per NumPy
    torch.manual_seed(seed)  # Imposta il seed per PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Imposta il seed per PyTorch (GPU)
        torch.cuda.manual_seed_all(seed)  # Se usi più GPU, imposta il seed per tutte

    # Imposta alcuni flag per garantire una piena riproducibilità (opzionali)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Funzione per allenare e valutare il modello

import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import torch
import joblib


# Definizione del modello con più strati e più neuroni
class FootballPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=3):
        super(FootballPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.softmax(self.fc_out(x))
        return x


import matplotlib.pyplot as plt


# Funzione per calcolare l'importanza delle feature tramite permutazione
def permutation_feature_importance(model, X_test, y_test, metric=accuracy_score, n_permutations=5, feature_names=None):
    model.eval()
    baseline_predictions = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    baseline_score = metric(y_test, baseline_predictions)

    importances = np.zeros(X_test.shape[1])
    accuracy_diffs = []

    for i in range(X_test.shape[1]):
        permuted_scores = []
        for _ in range(n_permutations):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, i])  # Permuta una singola colonna
            permuted_predictions = model(torch.tensor(X_permuted, dtype=torch.float32)).argmax(dim=1).numpy()
            permuted_score = metric(y_test, permuted_predictions)
            permuted_scores.append(permuted_score)

        # Calcola la differenza tra l'accuratezza baseline e quella permutata
        mean_permuted_score = np.mean(permuted_scores)
        accuracy_diff = baseline_score - mean_permuted_score
        importances[i] = accuracy_diff
        accuracy_diffs.append({
            'Feature': feature_names[i] if feature_names is not None else f'Feature_{i}',
            'Baseline Accuracy': baseline_score,
            'Permuted Accuracy': mean_permuted_score,
            'Accuracy Difference': accuracy_diff
        })

    # Creazione del DataFrame
    df_accuracy_diffs = pd.DataFrame(accuracy_diffs)

    # Salva il DataFrame in un file CSV
    df_accuracy_diffs.to_csv('./feature_importance_accuracy.csv', index=False)

    # Ordino i risultati per Accuratezza Differenza
    df_accuracy_diffs = df_accuracy_diffs.sort_values(by='Accuracy Difference', ascending=False)

    # Prendo solo le prime 20 feature più importanti
    df_top_20 = df_accuracy_diffs.head(20)

    # Creazione del grafico a barre orizzontale per le prime 20 feature
    plt.figure(figsize=(10, 8))
    plt.barh(df_top_20['Feature'], df_top_20['Accuracy Difference'], color="skyblue")
    plt.xlabel("Accuracy Difference")
    plt.ylabel("Features")
    plt.title("Top 20 Feature Importance Based on Accuracy Difference")
    plt.gca().invert_yaxis()  # Inverte l'asse y per visualizzare le feature più importanti in alto
    plt.tight_layout()

    # Salvo il grafico come immagine
    plt.savefig('./top_20_feature_importance_report.png')
    plt.show()

    return importances


# Funzione di allenamento con criteri, ottimizzatore e scheduler passati
def train_model(model, X_train, y_train, X_val, y_val, epochs, optimizer, scheduler, criterion):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break


# Funzione di valutazione
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()

        probs = outputs.numpy()

        y_test_one_hot = np.eye(3)[y_test]
        brier_score = brier_score_loss(y_test_one_hot.flatten(), probs.flatten())

        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        print(f'Brier Score: {brier_score:.4f}')

        return probs


# Funzione di train e valutazione con cross-validation
def train_and_evaluate_model_with_crossval(df, train, validation, test, epochs=1000, learning_rate=0.001, n_splits=5):
    set_seed(42)
    criterion = nn.CrossEntropyLoss()

    # Prepara i dati
    df_train, train_target, _ = cleaning_and_target(df[df['season'].isin(train)])
    df_test, test_target, odds = cleaning_and_target(df[df['season'].isin(test)])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train)
    X_test_scaled = scaler.transform(df_test)

    scaler_path = './salvo_models/scaler/scaler.joblib'
    joblib.dump(scaler, scaler_path)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_target.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_target.values, dtype=torch.long)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tensor, y_train_tensor)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_fold_train, X_fold_val = X_train_tensor[train_idx], X_train_tensor[val_idx]
        y_fold_train, y_fold_val = y_train_tensor[train_idx], y_train_tensor[val_idx]

        model = FootballPredictionModel(X_fold_train.shape[1], 256, 3)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-7)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        train_model(model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, epochs, optimizer, scheduler, criterion)

    final_model = FootballPredictionModel(X_train_tensor.shape[1], 512, 3)
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-7)
    final_scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=20, gamma=0.1)
    # Definizione della funzione di perdita (Cross Entropy Loss per classificazione multiclasse)


    train_model(final_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs, final_optimizer,
                final_scheduler, criterion)

    probs = evaluate(final_model, X_test_tensor, y_test_tensor)

    # Calcolo dell'importanza delle feature e creazione del file CSV
    feature_names = df_train.columns  # Usa i nomi delle colonne del DataFrame
    feature_importances = permutation_feature_importance(final_model, X_test_scaled, y_test_tensor.numpy(),
                                                         feature_names=feature_names)

    print("Feature Importances saved in 'feature_importance_accuracy.csv'")

    save_model_pth(final_model)
    return odds, probs, test_target


def save_model_pth(model):
    """Salva lo stato dei pesi del modello in un percorso specificato, numerando i file automaticamente."""
    model_dir = './salvo_models/models'
    os.makedirs(model_dir, exist_ok=True)  # Crea la directory se non esiste

    # Trova il primo numero disponibile per il nome del file
    model_files = os.listdir(model_dir)
    existing_numbers = [int(f.split('.')[0]) for f in model_files if f.split('.')[0].isdigit()]
    next_number = 0 if not existing_numbers else max(existing_numbers) + 1

    model_path = os.path.join(model_dir, f'{next_number}.pth')

    # Salva lo stato dei pesi del modello
    torch.save(model.state_dict(), model_path)
    print(f'Modello salvato come {model_path}')


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
    scaler_path = './salvo_models/scaler/scaler.joblib'

    # Carica lo scaler salvato
    scaler = joblib.load(scaler_path)

    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])  # Modifica per caricare modelli PyTorch (.pth)

    # Prepara il dataset di test
    df_test, results, odds = cleaning_and_target(df[df['season'].isin(test)])
    X_test = df_test

    # Usa lo scaler per trasformare i dati di test
    X_test_scaled = scaler.transform(X_test)  # Trasformazione con lo scaler salvato

    # Converte X_test_scaled in tensore PyTorch
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    results_list = []

    # Itera su ogni modello salvato
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)

        # Ricrea l'architettura del modello (assumendo che conosciamo input_size, hidden_size, output_size)
        model = FootballPredictionModel(input_size=X_test_tensor.shape[1], hidden_size=256, output_size=3)

        # Carica lo stato dei pesi del modello
        model.load_state_dict(torch.load(model_path, weights_only=True))

        # Metti il modello in modalità di valutazione
        model.eval()

        # Esegui le previsioni
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            #y_pred = outputs.argmax(dim=1).numpy()  # Predizione delle classi

        # Aggiunge i risultati alla lista
        results_list.append((odds, y_pred, results))

        # Stampa delle metriche per il modello corrente (puoi riattivare queste righe se ti servono)
        # print(f"Risultati per il modello {model_file}:")
        # mse = mean_squared_error(df_test[['prob_1', 'prob_X', 'prob_2']], y_pred)
        # r2 = r2_score(df_test[['prob_1', 'prob_X', 'prob_2']], y_pred)
        # print(f"Mean Squared Error: {mse}")
        # print(f"R^2 Score: {r2}\n")

    return results_list


def load_and_predict_all_models_(df, test):
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
    #X_test = df_test.drop(columns=['prob_1', 'prob_X', 'prob_2']).fillna(0)
    X_test = df_test
    #scaler = StandardScaler()
    #X_test_scaled = scaler.fit_transform(X_test)
    X_test_scaled = X_test
    results_list = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)

        # Aggiunge i risultati alla lista
        results_list.append((odds, y_pred, results))

        # Stampa delle metriche per il modello corrente
        #print(f"Risultati per il modello {model_file}:")
        #mse = mean_squared_error(df_test[['prob_1', 'prob_X', 'prob_2']], y_pred)
        #r2 = r2_score(df_test[['prob_1', 'prob_X', 'prob_2']], y_pred)
        #print(f"Mean Squared Error: {mse}")
        #print(f"R^2 Score: {r2}\n")

    return results_list