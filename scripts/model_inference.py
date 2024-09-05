import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss

from core.dataset.dataset_generation import generate_datasets


def classification_model_inference(data, dataset_params, class_model, go_live=False):
    model_probabilities = pd.DataFrame()
    # model_scores = pd.DataFrame()

    if go_live:
        dataset_params['drop_last_match_days'] = 0
        dataset_params['drop_first_match_days'] = 0
        dataset_params['drop_last_seasons'] = 0
        dataset_params['test_match_day'] = 0
        dataset_params['target_match_days'] = data[data['result_1X2'] == 'UNKNOWN']['match_day'].unique().tolist()

    datasets = generate_datasets(data, dataset_params)

    for i_day, dataset in datasets.items():
        x_target = dataset['target']['x']

        x_target = x_target.drop(['match_day', 'season'], axis=1)

        probabilities = class_model[i_day].predict_proba(x_target)

        prob_df = pd.concat((x_target, pd.DataFrame(probabilities, index=x_target.index)), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)

    target_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    sub_data = data[['league', 'season', 'match_day', 'match_n', 'Date', 'HomeTeam', 'AwayTeam',
                     'result_1X2', 'bet_1', 'bet_X', 'bet_2']]
    prediction = sub_data.merge(target_probabilities[['1', 'X', '2']], how='right', left_index=True, right_index=True)

    return prediction, target_probabilities
