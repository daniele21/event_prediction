import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss

from core.dataset.dataset_generation import generate_datasets


def classification_model_inference(data, dataset_params, class_model):
    model_probabilities = pd.DataFrame()
    model_scores = pd.DataFrame()

    datasets = generate_datasets(data, dataset_params)

    for i, dataset in datasets.items():
        x_target = dataset['target']['x']
        y_target = dataset['target']['y']

        predictions = class_model.predict(x_target)
        probabilities = class_model.predict_proba(x_target)
        recall = recall_score(y_target, predictions, average='weighted')
        precision = precision_score(y_target, predictions, average='weighted')
        f1 = f1_score(y_target, predictions, average='weighted')
        ll = log_loss(y_target, probabilities, labels=[0, 1, 2])

        scores = {'recall': recall,
                  'precision': precision,
                  'f1': f1,
                  'log_loss': ll,
                  }
        scores_df = pd.DataFrame(scores, index=[i])
        model_scores = pd.concat((model_scores, scores_df))

        prob_df = pd.concat((x_target, pd.DataFrame(probabilities, index=x_target.index)), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)

    target_scores = model_scores.reset_index() \
        .rename(columns={'index': 'target_day'})
    target_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    sub_data = data[['league', 'season', 'match_day', 'match_n', 'Date', 'HomeTeam', 'AwayTeam',
                     'result_1X2', 'bet_1', 'bet_X', 'bet_2']]
    prediction = sub_data.merge(target_probabilities[['1', 'X', '2']], how='right', left_index=True, right_index=True)

    return prediction, target_probabilities
