from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from core.grid_search import grid_search


def hyperparameter_tuning(datasets, model_params):

    estimator = model_params['estimator']
    param_grid = model_params['param_grid']
    scoring = model_params['scoring']

    performance_scores = grid_search(estimator, datasets, param_grid)
    best_params = performance_scores.drop(['target_day', 'recall',
                                           'precision', 'f1'], axis=1) \
        .drop_duplicates() \
        .sort_values(by=f'avg_{scoring}', ascending=False) \
        .drop(['estimator', 'avg_recall',
               'avg_precision', 'avg_f1'], axis=1) \
        .iloc[0] \
        .to_dict()

    return estimator, best_params


