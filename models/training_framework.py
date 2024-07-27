from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from core.grid_search import grid_search


def hyperparameter_tuning(datasets, model_params):
    estimator = model_params['estimator']
    param_grid = model_params['param_grid']

    performance_scores = grid_search(estimator, datasets, param_grid)
    best_params_dict = {}
    metrics = ['recall', 'precision', 'f1']
    for metric in metrics:
        scoring = f'avg_{metric}'

        best_params = performance_scores.drop(['target_day', 'recall',
                                               'precision', 'f1'], axis=1) \
            .drop_duplicates() \
            .sort_values(by=scoring, ascending=False) \
            .drop(['estimator', 'avg_recall',
                   'avg_precision', 'avg_f1'], axis=1) \
            .iloc[0] \
            .to_dict()

        best_params_dict[metric] = best_params

    return estimator, best_params_dict, performance_scores
