from lib.screapig_csv import features, filter_csv, get_first_encounters, get_second_encounters, process_results, \
    csv_validate_models_exact, csv_validate_models_double, calculate_accuracy, calculate_gains_losses, \
    kelly_calculate_gains_losses, best_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib
matplotlib.use('Agg')  # Imposta il backend non interattivo
def plot_ROC(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # Crea una lista di soglie da 0.1 a 1 con passo di 0.1
    threshold_list = [x/10 for x in range(1, 11)]

    # Plot della curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')  # Linea di riferimento per il caso casuale
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Plot delle soglie
    for threshold in threshold_list:
        plt.scatter(fpr[thresholds >= threshold][-1], tpr[thresholds >= threshold][-1], c='red')
        plt.annotate(f'Thsd: {threshold}', (fpr[thresholds >= threshold][-1], tpr[thresholds >= threshold][-1]),
                     textcoords="offset points", xytext=(-10,10), ha='center')

    # Calcola la distanza di Youden's J per ciascuna soglia
    youden_j = tpr - fpr

    # Trova l'indice della soglia che massimizza Youden's J
    best_threshold_index = np.argmax(youden_j)

    # Ottieni la migliore soglia
    best_threshold = thresholds[best_threshold_index]

    if best_threshold > 1:
        best_threshold = best_threshold - 1
    print("Best threshold:", best_threshold)

    plt.savefig(f'./predictions_file/roc_curve.png')  # Salva il grafico in un file
    plt.close()

    return best_threshold

# Funzione per codificare la colonna result_1X2
def encode_result(result):
    if result == '1':
        return 1
    else:
        return 0

# Funzione per la cross-validazione e selezione del miglior modello
def prob_train(pipe, X, y):
    best_model = None
    best_score = 0
    best_proba = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')

        if score > best_score:
            best_model = pipe
            best_score = score

    best_proba = best_model.predict_proba(X)
    return best_model, best_proba

# Funzione per eseguire la Grid Search per ottimizzare gli iperparametri
def grid_search_optimization(X, y):
    model = RandomForestClassifier()

    # Definire la pipeline
    estimators = [
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest()),
        ('classifier', model)
    ]
    RFpipe = Pipeline(estimators)

    # Definire i parametri per la Grid Search
    param_grid = {
        'feature_selection__k': [1, 2, 5, 10],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 12, 15],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # Definire la Grid Search con StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=RFpipe, param_grid=param_grid, scoring='f1_weighted', cv=skf, n_jobs=-1)

    # Eseguire la Grid Search
    grid_search.fit(X, y)

    # Miglior pipeline trovata dalla Grid Search
    best_pipe = grid_search.best_estimator_
    best_score = grid_search.best_score_

    # Eseguire la funzione prob_train con la migliore pipeline e il dataset completo
    best_pipe, best_proba = prob_train(best_pipe, X, y)

    return best_pipe, best_proba, best_score

def main():
    file_name = "serie_a_npm=5.csv"
    feature_list = features
    df = filter_csv(file_name, feature_list, start_dir='../tests')

    # Apply the encoding
    df['result_encoded'] = df['result_1X2'].apply(encode_result)

    # Drop rows with NaN values
    df = df.dropna()

    # Split the data by season
    train_data = df[df['season'].isin([0, 1, 2, 3, 4])]
    test_data = df[df['season'].isin([5])]

    columns_to_drop = ['HomeTeam', 'AwayTeam', 'result_1X2', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points', 'point_diff', 'goals_diff']

    X = train_data.drop(columns=columns_to_drop)
    y = train_data['result_encoded']
    best_pipe, best_proba, best_score = grid_search_optimization(X, y)

    # Estrazione delle probabilità per ogni classe
    best_proba = best_proba[:, 1]  # Probabilità della vittoria della squadra di casa

    best_threshold = plot_ROC(y, best_proba)  # Supponendo che plot_ROC accetti le probabilità della classe 1

    predictions = (best_proba >= best_threshold).astype(int)

    # Output dei risultati
    print("Best Model:", best_pipe)
    print("Best Probabilities for Home Win:", best_proba)
    print("Best F1 Score:", best_score)
    print("Classification Report:\n", classification_report(y, predictions))



def test():
    RandomForest = RandomForestClassifier(max_depth=10, n_estimators=200, min_samples_split=6)
    RFfeature_selection__k = 5

    estimators = [('scaler', StandardScaler()), ('feature_selection', SelectKBest(k=RFfeature_selection__k)),
                  ('classifier', RandomForest)]
    RFpipe = Pipeline(estimators)
    RFpipe.fit(X, y)


    # Estrai la colonna corrispondente all'indice selected_feature_index
    selected_column = X.iloc[:, selected_feature_index]
    selected_column_df = pd.DataFrame(selected_column)

    # RFpipe.fit(X_train, y_train)
    RFpipe, RFprob = prob_train(RFpipe, selected_column_df, y)
if __name__ == "__main__":
    main()
