import pandas as pd

from core.grid_search import grid_search
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



