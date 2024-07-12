# Nome del Progetto

Questo progetto fornisce varie funzioni per filtrare e processare dati di partite di calcio, calcolare probabilità e validare modelli predittivi. Di seguito sono riportate le definizioni e i parametri per ciascuna funzione nel progetto.

## Funzioni

### 1. `filter_csv`

Filtra il file CSV in base alle caratteristiche fornite e alla directory di partenza.

**Parametri:**
- `file_name` (str): Il nome del file CSV.
- `feature_list` (list): Un elenco di caratteristiche per il filtraggio.
- `start_dir` (str): La directory di partenza per cercare il file CSV.

**Ritorna:**
- `pd.DataFrame`: Il DataFrame filtrato.

### 2. `get_first_encounters`

Restituisce un DataFrame contenente solo il primo incontro tra ciascuna coppia di squadre, quindi metà campionato.

**Parametri:**
- `df` (pd.DataFrame): Il DataFrame originale contenente le partite.

**Ritorna:**
- `pd.DataFrame`: Un nuovo DataFrame con solo il primo incontro tra ciascuna coppia di squadre.

### 3. `get_second_encounters`

Restituisce un DataFrame contenente solo il secondo incontro tra ciascuna coppia di squadre, quindi la seconda e utlima parte del campionato.

**Parametri:**
- `df` (pd.DataFrame): Il DataFrame originale contenente le partite.

**Ritorna:**
- `pd.DataFrame`: Un nuovo DataFrame con solo il secondo incontro tra ciascuna coppia di squadre.

### 4. `process_results`

Elabora i risultati delle partite e restituisce una struttura JSON con le statistiche per ciascuna squadra.

**Parametri:**
- `df` (pd.DataFrame): Il DataFrame contenente i dati delle partite.

**Ritorna:**
- `dict`: Un dizionario con le statistiche per ciascuna squadra.
**Struttura del dizionario restituito:**
```python
{
    'home': {
        'tot_match': 0,
        'wins': 0,
        'draws': 0,
        'losses': 0
    },
    'away': {
        'tot_match': 0,
        'wins': 0,
        'draws': 0,
        'losses': 0
    }
}
```
### 5. `csv_validate_models_exact`

Controlla se la probabilità più alta tra 1,x e 2 corrisponde all'esito effettivo della partità e crea un csv per poter confrontare con la realtà

**Parametri:**
- `file_name` (str): Il nome del file CSV.
- `teamA` (str): Il nome della squadra di casa.
- `teamB` (str): Il nome della squadra ospite.
- `A_win_prob` (float): La probabilità di vittoria della squadra di casa.
- `A_draw_prob` (float): La probabilità di pareggio.
- `A_loss_prob` (float): La probabilità di sconfitta della squadra di casa.
- `ground_truth` (str): Il risultato reale della partita.
- `quote_home` (float): La quota della vittoria della squadra di casa.
- `quote_draw` (float): La quota del pareggio.
- `quote_way` (float): La quota della vittoria della squadra ospite.

**Ritorna:**
- Nessuno

### 6. `csv_validate_models_double`

Controlla se la probabilità più alta tra 1X,X2 e 12 corrisponde all'esito effettivo della partità e crea un csv per poter confrontare con la realtà

**Parametri:**
- `file_name` (str): Il nome del file CSV.
- `teamA` (str): Il nome della squadra di casa.
- `teamB` (str): Il nome della squadra ospite.
- `A_win_prob` (float): La probabilità di vittoria della squadra di casa.
- `A_draw_prob` (float): La probabilità di pareggio.
- `A_loss_prob` (float): La probabilità di sconfitta della squadra di casa.
- `ground_truth` (str): Il risultato reale della partita.
- `quote_home` (float): La quota della vittoria della squadra di casa.
- `quote_draw` (float): La quota del pareggio.
- `quote_way` (float): La quota della vittoria della squadra ospite.

**Ritorna:**
- Nessuno

### 7. `calculate_accuracy`

Calcola l'accuratezza del modello basata sui risultati nel file CSV.

**Parametri:**
- `file_name` (str): Il nome del file CSV.

**Ritorna:**
- `float`: L'accuratezza del modello.

### 8. `calculate_gains_losses`

Calcola i guadagni e le perdite totali basati sui risultati nel file CSV.

**Parametri:**
- `file_path` (str): Il percorso del file CSV.
- `amount` (float): L'importo della puntata unitaria.

**Ritorna:**
- Nessuno

## Caratteristiche

Le seguenti caratteristiche sono utilizzate per filtrare ed elaborare i dati delle partite:
- `Unnamed: 0`
- `league`
- `season`
- `match_n`
- `Date`
- `HomeTeam`
- `AwayTeam`
- `home_goals`
- `away_goals`
- `result_1X2`
- `bet_1`
- `bet_X`
- `bet_2`
- `home_points`
- `away_points`
- `cum_home_points`
- `cum_away_points`
- `home_league_points`
- `away_league_points`
- `cum_home_goals`
- `cum_away_goals`
- `home_league_goals`
- `away_league_goals`
- `point_diff`
- `goals_diff`
- `HOME_last-1-away`
- `HOME_last-1`
- `AWAY_last-1-home`
- `AWAY_last-1`
- `AWAY_last-1-away`
- `AWAY_last-2`
- `HOME_last-1-home`
- `HOME_last-2`
- `HOME_last-2-away`
- `HOME_last-3`
- `AWAY_last-2-home`
- `AWAY_last-3`
- `HOME_last-2-home`
- `HOME_last-4`
- `AWAY_last-2-away`
- `AWAY_last-4`
- `HOME_last-3-away`
- `HOME_last-5`
- `AWAY_last-3-home`
- `AWAY_last-5`
- `HOME_last-3-home`
- `AWAY_last-3-away`
- `HOME_last-4-away`
- `AWAY_last-4-home`
- `HOME_last-4-home`
- `AWAY_last-4-away`
- `HOME_last-5-away`
- `AWAY_last-5-home`
- `HOME_last-5-home`
- `AWAY_last-5-away`
