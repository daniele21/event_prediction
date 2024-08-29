from itertools import product

import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from core.utils import get_estimator


def grid_search(estimator, datasets, param_grid):
    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    final_scores = pd.DataFrame()
    model_probabilities = pd.DataFrame()

    for params in tqdm(param_combinations, total=len(param_combinations), desc=' > Tuning Hyperparameters: '):
        params_scores = pd.DataFrame()
        model = estimator(**params)

        for i, dataset in datasets.items():
            x_train, x_test = dataset['train']['x'], dataset['test']['x']
            y_train, y_test = dataset['train']['y'], dataset['test']['y']

            # Create and train the model
            if estimator.__name__ in 'LGBMClassifier':
                model.fit(x_train, y_train.squeeze(),
                          eval_set=[(x_train, y_train), (x_test, y_test)])
            else:
                model.fit(x_train, y_train.squeeze())

            # Evaluate the model
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)
            recall = recall_score(y_test, predictions, average='weighted')
            precision = precision_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            ll = log_loss(y_test, probabilities)

            scores = {'estimator': estimator.__name__,
                      **params,
                      'recall': recall,
                      'precision': precision,
                      'f1': f1,
                      'log_loss': ll}
            scores_df = pd.DataFrame(scores, index=[i])
            params_scores = pd.concat((params_scores, scores_df))

            prob_df = pd.concat((x_test, pd.DataFrame(probabilities, index=x_test.index)), axis=1)
            model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)

        for x in ['recall', 'precision', 'f1', 'log_loss']:
            params_scores[f'avg_{x}'] = params_scores[x].mean()

        final_scores = pd.concat((final_scores, params_scores))

    final_scores = final_scores.reset_index().rename(columns={'index': 'target_day'})
    model_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    return final_scores, model_probabilities


def grid_search_regression(dataset, strategy_params):
    estimator = get_estimator(strategy_params['estimator'])
    param_grid = strategy_params['param_grid']
    scoring = strategy_params['scoring']
    cv_fold = strategy_params['cv_fold']

    x_train, y_train = dataset['train']['x'], dataset['train']['y']
    x_test, y_test = dataset['test']['x'], dataset['test']['y']

    grid_search = GridSearchCV(estimator=estimator(),
                               param_grid=param_grid,
                               cv=cv_fold,
                               scoring=scoring,
                               n_jobs=-1,
                               verbose=1)

    # Perform grid search on the filtered training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters and the best score from the grid search
    best_params = grid_search.best_params_

    return estimator, best_params