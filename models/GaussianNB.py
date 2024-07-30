from lib.screapig_csv import features, filter_csv, get_first_encounters, get_second_encounters, process_results, \
    csv_validate_models_exact, csv_validate_models_double, calculate_accuracy, calculate_gains_losses, \
    kelly_calculate_gains_losses, best_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Encoding the result_1X2 column
def encode_result(result):
    if result == '1':
        return 1
    elif result == '2':
        return 0
    elif result == 'X':
        return 2

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

    columns_to_drop = ['result_1X2', 'result_encoded', 'Unnamed: 0', 'Date', 'league',
                       'home_goals', 'away_goals', 'away_points', 'bet_1', 'bet_X', 'bet_2',
                       'season', 'match_n', 'home_points', 'point_diff', 'goals_diff']

    # Filtra il DataFrame con le colonne da rimuovere
    X_train = train_data.drop(columns=columns_to_drop)
    y_train = train_data['result_encoded']
    X_test = test_data.drop(columns=columns_to_drop)
    y_test = test_data['result_encoded']

    # Encode categorical features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align columns of test set to match training set
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Train the Gradient Boosting Classifier
    model = GaussianNB()


    model.fit(X_train, y_train)

    # Prepare a list to store the predictions
    predictions = []

    # Iterate over each test match
    # Prepare a list to store the predictions
    predictions = []

    # Iterate over each test match
    for index, row in X_test.iterrows():
        match_features = row.to_frame().T

        # Get the probability predictions for the match
        match_proba = model.predict_proba(match_features)[0]
        match_pred = model.predict(match_features)[0]

        home_team = test_data.loc[index, 'HomeTeam']
        away_team = test_data.loc[index, 'AwayTeam']
        home_win_prob = match_proba[1]
        draw_prob = match_proba[2]
        away_win_prob = match_proba[0]
        quote_home = test_data.loc[index, 'bet_1']
        quote_draw = test_data.loc[index, 'bet_X']
        quote_way = test_data.loc[index, 'bet_2']

        # Append the predictions to the list
        predictions.append(match_pred)

        csv_validate_models_exact('GaussianNB_exact.csv', home_team, away_team, home_win_prob,
                                  draw_prob, away_win_prob, test_data.loc[index, 'result_1X2'], quote_home,
                                  quote_draw, quote_way)
        csv_validate_models_double('GaussianNB_double.csv', home_team, away_team, home_win_prob,
                                   draw_prob, away_win_prob, test_data.loc[index, 'result_1X2'], quote_home,
                                   quote_draw, quote_way)

    # Calculate the evaluation metrics
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
    print(f"gradient_boosting_exact: {calculate_accuracy('GaussianNB_exact.csv')}")
    print(f"gradient_boosting_double: {calculate_accuracy('GaussianNB_double.csv')}")
    print("Total gain with exact results")
    value = 10
    min_prob, min_gain = -1 , -1
    #min_prob, min_gain = best_model(value, "gradient_boosting_exact.csv", "gradient_boosting_double.csv")
    gain = calculate_gains_losses("GaussianNB_exact.csv", value, min_prob, min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('GaussianNB_exact.csv')}")

    print("\nTotal gain with double results")
    gain = calculate_gains_losses("GaussianNB_double.csv", value, min_prob, min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('GaussianNB_double.csv')}")

    print("\nTotal gain with exact results with kelly")
    gain = kelly_calculate_gains_losses("GaussianNB_exact.csv", value, min_prob, min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('GaussianNB_exact.csv')}")

    print("\nTotal gain with double results with kelly")
    gain = kelly_calculate_gains_losses("GaussianNB_double.csv", value, min_prob, min_gain)
    print(f"accuracy_prediction: {calculate_accuracy('GaussianNB_double.csv')}")

