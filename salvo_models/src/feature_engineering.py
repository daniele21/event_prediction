import pandas as pd
import numpy as np
import os

def expand_features(df):
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

    df['HomeTeam_mean_goals'] = rolling_mean_goals(df, 'HomeTeam', 'home_goals')
    df['AwayTeam_mean_goals'] = rolling_mean_goals(df, 'AwayTeam', 'away_goals')

    # 2. Differenza di forza tra le squadre basata su goal segnati
    df['Goals_diff'] = df['HomeTeam_mean_goals'] - df['AwayTeam_mean_goals']

    # 3. Effetto casa (percentuale di vittorie in casa)
    df['HomeTeam_win_percent'] = df.groupby('HomeTeam')['home_goals'].transform(
        lambda x: (x > 0).rolling(10, min_periods=1).mean())


    # 4. Momento della stagione (esempio basato su match day)
    df['Season_phase'] = pd.cut(df['match_day'], bins=[0, 10, 25, 38], labels=[0, 1, 2])

    # 5. Creazione di interazione tra quote e performance recente
    if 'Quote_1' in df.columns and 'Quote_2' in df.columns:
        df['Quote_Performance_interaction'] = (df['Quote_1'] - df['Quote_2']) * df['Goals_diff']

    # 6. Interazioni tra statistiche (esempio)
    df['Home_Away_goal_interaction'] = df['HomeTeam_mean_goals'] * df['AwayTeam_mean_goals']

    return df


# Esempio di utilizzo in un altro file:
if __name__ == "__main__":
    # Carica il dataset
    df = pd.read_csv('path_to_your_file.csv')

    # Chiama la funzione per espandere il dataset
    df_expanded = expand_features(df)

    # Mostra le prime righe del DataFrame espanso
    print(df_expanded.head())
