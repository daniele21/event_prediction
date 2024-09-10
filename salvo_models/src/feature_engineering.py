import pandas as pd
import numpy as np
import os

def expand_features_(df):
    """
    Espande il dataset aggiungendo nuove feature utili per la previsione di risultati sportivi.

    Args:
    df (pd.DataFrame): Il DataFrame originale che contiene i dati delle partite.

    Returns:
    pd.DataFrame: Il DataFrame aggiornato con nuove feature aggiunte.
    """

    # 1. Media dei goal segnati e subiti nelle ultime 5 partite per la squadra di casa e in trasferta
    def rolling_mean_goals(df, team_col, goals_col, window=5):
        return df.groupby(team_col)[goals_col].transform(lambda x: x.rolling(window, min_periods=1).mean())

   #df['HomeTeam_mean_goals'] = rolling_mean_goals(df, 'HomeTeam', 'home_goals')
    #df['AwayTeam_mean_goals'] = rolling_mean_goals(df, 'AwayTeam', 'away_goals')

    # 2. Differenza di forza tra le squadre basata su goal segnati
    #df['Goals_diff'] = df['HomeTeam_mean_goals'] - df['AwayTeam_mean_goals']

    # 3. Effetto casa (percentuale di vittorie in casa)
    #df['HomeTeam_win_percent'] = df.groupby('HomeTeam')['home_goals'].transform(
        #lambda x: (x > 0).rolling(10, min_periods=1).mean())


    # 4. Momento della stagione (esempio basato su match day)
    df['Season_phase'] = pd.cut(df['match_day'], bins=[0, 10, 25, 38], labels=[0, 1, 2])

    # 5. Creazione di interazione tra quote e performance recente
    #if 'Quote_1' in df.columns and 'Quote_2' in df.columns:
        #df['Quote_Performance_interaction'] = (df['Quote_1'] - df['Quote_2']) * df['Goals_diff']

    # 6. Interazioni tra statistiche (esempio)
    #df['Home_Away_goal_interaction'] = df['HomeTeam_mean_goals'] * df['AwayTeam_mean_goals']

    return df


def expand_features(df, columns_to_drop):
    """
    Funzione che genera nuove feature statistiche avanzate per il dataframe, senza utilizzare le colonne
    specificate in columns_to_drop. Le feature includono medie mobili, media esponenziale, derivate,
    cumulativi, percentuali, e altre feature.

    Args:
    df (pd.DataFrame): Il dataframe originale.
    columns_to_drop (list): Lista delle colonne da escludere dalla creazione di nuove feature.

    Returns:
    pd.DataFrame: Un nuovo dataframe con le nuove feature aggiunte.
    """
    # Crea una copia del dataframe originale
    df_ = df.copy()

    # Rimuovi le colonne specificate (errors='ignore' eviterà errori se una colonna non esiste)
    df_clean = df_.drop(columns=columns_to_drop, errors='ignore')

    # Seleziona solo le colonne numeriche presenti nel dataframe pulito
    numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns

    # Crea un dizionario per contenere le nuove feature
    new_features_dict = {}

    # Per ogni colonna numerica, genera nuove feature statistiche avanzate
    for col in numerical_cols:
        # Media mobile (finestra di 3 e 5)
        new_features_dict[f'{col}_rolling_mean_3'] = df_clean[col].rolling(window=3, min_periods=1).mean()
        new_features_dict[f'{col}_rolling_mean_5'] = df_clean[col].rolling(window=5, min_periods=1).mean()

        # Media esponenziale (span 3 e 5)
        new_features_dict[f'{col}_ewm_3'] = df_clean[col].ewm(span=3, adjust=False).mean()
        new_features_dict[f'{col}_ewm_5'] = df_clean[col].ewm(span=5, adjust=False).mean()

        # Prime e seconde derivate (differenze)
        new_features_dict[f'{col}_diff_1'] = df_clean[col].diff(1)
        new_features_dict[f'{col}_diff_2'] = df_clean[col].diff(2)

        # Deviazione standard (finestra di 3 e 5)
        new_features_dict[f'{col}_rolling_std_3'] = df_clean[col].rolling(window=3, min_periods=1).std()
        new_features_dict[f'{col}_rolling_std_5'] = df_clean[col].rolling(window=5, min_periods=1).std()

        # Somma cumulativa
        new_features_dict[f'{col}_cumsum'] = df_clean[col].cumsum()

    # Crea un DataFrame da tutte le nuove feature generate
    new_features_df = pd.DataFrame(new_features_dict)

    # Combina il dataframe originale con il nuovo dataframe delle feature
    return pd.concat([df_, new_features_df], axis=1)
# Esempio di utilizzo in un altro file:

def expand_features(df, columns_to_drop):
    """
    Funzione che genera nuove feature statistiche avanzate per il dataframe, senza utilizzare le colonne
    specificate in columns_to_drop. Le feature includono medie mobili, media esponenziale, derivate,
    cumulativi, percentuali, e altre feature.

    Args:
    df (pd.DataFrame): Il dataframe originale.
    columns_to_drop (list): Lista delle colonne da escludere dalla creazione di nuove feature.

    Returns:
    pd.DataFrame: Un nuovo dataframe con le nuove feature aggiunte.
    """
    # Crea una copia del dataframe originale
    df_ = df.copy()

    # Rimuovi le colonne specificate (errors='ignore' eviterà errori se una colonna non esiste)
    df_clean = df_.drop(columns=columns_to_drop, errors='ignore')

    # Combina il dataframe originale con il nuovo dataframe delle feature
    return df_clean
# Esempio di utilizzo in un altro file:
if __name__ == "__main__":
    # Carica il dataset
    df = pd.read_csv('path_to_your_file.csv')

    # Chiama la funzione per espandere il dataset
    df_expanded = expand_features(df)

    # Mostra le prime righe del DataFrame espanso
    print(df_expanded.head())
