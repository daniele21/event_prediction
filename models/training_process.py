import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm

from scripts.training.validation import check_calibration


def training_process(datasets, estimator, params, inference=False):
    model_scores = pd.DataFrame()
    model_probabilities = pd.DataFrame()
    model = estimator(**params)

    for i, dataset in tqdm(datasets.items(),
                           desc=' > Training Steps: '):
        x_train, x_test, x_target = dataset['train']['x'], dataset['test']['x'], dataset['target']['x']
        y_train, y_test, y_target = dataset['train']['y'], dataset['test']['y'], dataset['target']['y']

        if inference:
            x_train = pd.concat((x_train, x_test), axis=0)
            y_train = pd.concat((y_train, y_test), axis=0)

        # Create and train the model
        if estimator.__name__ == 'LGBMClassifier':
            model.fit(x_train, y_train.squeeze(),
                      eval_set=[(x_train, y_train), (x_test, y_test)])
        else:
            model.fit(x_train, y_train.squeeze())

        # Evaluate the model
        x_valid_set = x_target if inference else x_test
        y_valid_set = y_target if inference else y_test
        predictions = model.predict(x_valid_set)
        probabilities = model.predict_proba(x_valid_set)
        recall = recall_score(y_valid_set, predictions, average='weighted')
        precision = precision_score(y_valid_set, predictions, average='weighted')
        f1 = f1_score(y_valid_set, predictions, average='weighted')

        scores = {'estimator': estimator.__name__,
                  **params,
                  'recall': recall,
                  'precision': precision,
                  'f1': f1
                  }
        scores_df = pd.DataFrame(scores, index=[i])
        model_scores = pd.concat((model_scores, scores_df))

        prob_df = pd.concat((x_valid_set, pd.DataFrame(probabilities, index=x_valid_set.index)), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)

    model_scores = model_scores.reset_index()\
                                .rename(columns={'index': 'target_day'})
    model_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    return model, model_scores, model_probabilities
