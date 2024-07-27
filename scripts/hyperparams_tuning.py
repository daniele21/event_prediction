import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import config.league as LEAGUE
from config.constants import MATCH_RESULT_V1
from core.dataset.dataset_generation import generate_datasets
from core.utils import get_timestamp, ensure_folder, get_most_recent_data
from models.training_framework import hyperparameter_tuning


def tuning(data, dataset_params, model_params):

    datasets = generate_datasets(data, dataset_params)
    estimator, best_params_dict, perf_scores = hyperparameter_tuning(datasets, model_params)

    for key, value in dataset_params.items():
        perf_scores.insert(0, key, str(value))

    return estimator, best_params_dict, perf_scores


if __name__ == '__main__':
    league_name = LEAGUE.PREMIER_2
    npm = 5
    test_name = 'target_2324'

    league_dir = f'resources/{league_name}/'
    source_data = get_most_recent_data(league_dir, league_name, n_prev_match=npm)

    dataset_params = {'drop_last': 5,
                      'drop_first': 5,
                      'last_n_seasons': 5,
                      'target_match_days': np.arange(7, 31),
                      'test_match_day': 2,
                      'preprocessing_version': MATCH_RESULT_V1,
                      }

    param_grid = {
        'n_estimators': [300],  # Number of trees in the forest
        'max_depth': [5, 10, 20, 30],  # Maximum number of levels in each decision tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        'bootstrap': [False],  # Method for sampling data points (with or without replacement)
        'random_state': [2024]
    }

    model_params = {'estimator': RandomForestClassifier,
                    'param_grid': param_grid}

    data = pd.read_csv(source_data,
                       index_col=0)

    estimator, best_params, perf_scores = tuning(data, dataset_params, model_params)

    perf_scores.insert(0, 'source', source_data)
    output_dir = f'outputs/tuning/{league_name}/'
    ensure_folder(output_dir)
    perf_scores.to_csv(f'{output_dir}{test_name}_fine_tuning_{get_timestamp()}.csv')
