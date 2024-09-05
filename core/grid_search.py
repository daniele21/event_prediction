from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core.strategies.custom_loss import profit_based_score, custom_scorer
from core.utils import get_estimator


def ts_grid_search(estimator, datasets, param_grid,
                   splits=5, max_train_size=5*38*10, test_size=5*10):
    tscv = TimeSeriesSplit(n_splits=splits,
                           max_train_size=max_train_size,
                           test_size=test_size)

    best_params_dict = {}
    results = pd.DataFrame()
    for i_day, dataset in tqdm(datasets.items()):
        x_train, x_test = dataset['train']['x'], dataset['test']['x']
        y_train, y_test = dataset['train']['y'], dataset['test']['y']

        x_train = x_train.drop(['match_day', 'season'], axis=1)
        x_test = x_test.drop(['match_day', 'season'], axis=1)

        grid_search = GridSearchCV(estimator=estimator(),
                                   param_grid=param_grid,
                                   cv=tscv,
                                   scoring='neg_log_loss',
                                   n_jobs=-1,
                                   verbose=1,
                                   error_score='raise')

        if estimator.__name__ == 'LGBMClassifier':
            fit_params = {'eval_set': [(x_test, y_test)]}
        else:
            fit_params = {}

        # Fit GridSearchCV
        grid_search.fit(x_train, y_train, **fit_params)
        best_params = grid_search.best_params_

        if estimator.__name__ in 'LGBMClassifier':
            num_iterations = grid_search.best_estimator_._best_iteration
            best_params['num_iterations'] = num_iterations

        test_results = pd.DataFrame()
        for i in range(splits):
            feature = f'split{i}_test_score'
            test_result = pd.DataFrame(grid_search.cv_results_[feature], columns=[feature])
            test_results = test_results.merge(test_result,
                                              left_index=True,
                                              right_index=True,
                                              how='outer')
        cv_params = pd.DataFrame(grid_search.cv_results_['params'])
        test_results = test_results.merge(cv_params,
                           left_index=True,
                           right_index=True,
                           )
        test_results['target_day'] = i_day
        best_params_dict[i_day] = best_params

    results = pd.concat((results, test_results))

    return best_params_dict, results


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

            x_train = x_train.drop(['match_day', 'season'], axis=1)
            x_test = x_test.drop(['match_day', 'season'], axis=1)
            # x_target = x_target.drop(['match_day', 'season'], axis=1)

            # Create and train the model
            if estimator.__name__ in 'LGBMClassifier':
                model.fit(x_train, y_train.squeeze(),
                          eval_set=[(x_train, y_train), (x_test, y_test)])
                num_iterations = model._best_iteration
                params['num_iterations'] = num_iterations
            else:
                model.fit(x_train, y_train.squeeze())

            print(params)

            # Evaluate the model
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)
            recall = recall_score(y_test, predictions, average='weighted')
            precision = precision_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            ll = -log_loss(y_test, probabilities)

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
    if estimator.__name__ == 'LGBMRegressor':
        fit_params = {'eval_set': [(x_train, y_train), (x_test, y_test)]}
    else:
        fit_params = {}

    if estimator.__name__ in ('LinearRegression', 'ElasticNet'):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Perform grid search on the filtered training data
    grid_search.fit(x_train, y_train.squeeze(), **fit_params)

    # Get the best parameters and the best score from the grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_model, best_params