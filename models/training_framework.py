import pandas as pd

from core.grid_search import grid_search, ts_grid_search
from core.utils import get_estimator


def hyperparameter_tuning(datasets, model_params):
    estimator = get_estimator(model_params['estimator'])
    param_grid = model_params['param_grid']

    performance_scores, probs = grid_search(estimator, datasets, param_grid)
    best_params_dict = {}
    metrics = ['recall', 'precision', 'f1', 'log_loss']
    for metric in metrics:
        scoring = f'avg_{metric}'
        ascending = False if metric != 'log_loss' else True
        best_params = performance_scores.drop(['target_day', 'recall',
                                               'precision', 'f1', 'log_loss'], axis=1) \
            .drop_duplicates() \
            .sort_values(by=scoring, ascending=ascending) \
            .drop(['estimator', 'avg_recall',
                   'avg_precision', 'avg_f1', 'avg_log_loss'], axis=1) \
            .iloc[0] \
            .to_dict()

        best_params_dict[metric] = best_params

    best_params_df = pd.DataFrame(best_params_dict).T

    return estimator, best_params_df, performance_scores


def new_new_hyperparameter_tuning(datasets, model_params):
    estimator = get_estimator(model_params['estimator'])
    param_grid = model_params['param_grid']
    splits = model_params['splits']
    max_train_size = model_params['max_train_size']
    test_size = model_params['test_size']

    # performance_scores, probs = grid_search(estimator, datasets, param_grid)
    best_params_dict, performance_scores = ts_grid_search(estimator, datasets, param_grid,
                                               splits=splits,
                                               max_train_size=max_train_size,
                                               test_size=test_size)

    return best_params_dict, performance_scores

def new_hyperparameter_tuning(datasets, model_params):
    estimator = get_estimator(model_params['estimator'])
    param_grid = model_params['param_grid']

    performance_scores, probs = grid_search(estimator, datasets, param_grid)
    performance_scores = performance_scores.sort_values(by=['target_day'])

    best_params_dict = {}
    metrics = ['precision', 'f1', 'recall', 'log_loss']
    for metric in metrics:
        other_metrics = [x for x in metrics if x != metric]
        sort_order = [metric] + other_metrics
        best_params_dict[metric] = {}
        for target_day in performance_scores['target_day'].unique():
            best_params = performance_scores[performance_scores['target_day'] == target_day]\
                                .sort_values(by=sort_order,
                                             ascending=False)
            best_params = best_params.drop(['target_day', 'estimator', 'avg_recall',
                                            'avg_precision', 'avg_f1', 'avg_log_loss'], axis=1)
            best_params = best_params.drop(metrics, axis=1)
            best_params_dict[metric][int(target_day)] = best_params.iloc[0].to_dict()

    return estimator, best_params_dict, performance_scores
