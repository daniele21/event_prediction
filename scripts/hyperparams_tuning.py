import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
import config.league as LEAGUE
from config.constants import MATCH_RESULT_V1
from core.dataset.dataset_generation import generate_datasets
from core.utils import get_timestamp, ensure_folder, get_most_recent_data
from models.training_framework import hyperparameter_tuning


def tuning(data, dataset_params, model_params):

    datasets = generate_datasets(data, dataset_params)
    estimator, best_params_df, perf_scores = hyperparameter_tuning(datasets, model_params)

    for key, value in dataset_params.items():
        perf_scores.insert(0, key, str(value))

    return estimator, best_params_df, perf_scores


if __name__ == '__main__':
    league_name = LEAGUE.SERIE_A
    npm = 5
    test_name = 'target_2324'

    league_dir = f'resources/{league_name}/'
    source_data = get_most_recent_data(league_dir, league_name, n_prev_match=npm)

    dataset_params = {'drop_last': 5,
                      'drop_first': 5,
                      'last_n_seasons': 13,
                      'target_match_days': np.arange(9,31),
                      'test_match_day': 2,
                      'preprocessing_version': MATCH_RESULT_V1,
                      }
    # RF
    # param_grid = {
    #     'n_estimators': [300],  # Number of trees in the forest
    #     'max_depth': [10, 30],  # Maximum number of levels in each decision tree
    #     'min_samples_split': [2, 5],  # Minimum number of samples required to split a node
    #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    #     #'bootstrap': [False],  # Method for sampling data points (with or without replacement)
    #     'criterion': ["gini", "entropy", "log_loss"],
    #     'class_weight': [None, 'balanced'],
    #     'random_state': [2024],
    #     'n_jobs': [-1]
    # }

    # LGBM
    param_grid = {
        'num_leaves': [5, 10, 15],  # [10, 20, 30],
        'max_depth': [5, 10, 30],  # [5, 10, 20, 45],
        # 'min_data_in_leaf': [10, 20, 30],
        'feature_fraction': [0.8, 1],
        'learning_rate': [0.1, 0.5, 0.01],
        'n_estimators': [300],
        # 'lambda_l1': [0, 0.2],
        # 'lambda_l2': [0, 0.2],
        # 'num_iteration': [50000],
        # 'early_stopping_round': [10, 30],
        'class_weight': ['balanced', None],
        'deterministic': [True],
        'seed': [2024]
    }

    model_params = {
                    'estimator': lgbm.LGBMClassifier,
                    # 'estimator': RandomForestClassifier,
                    'param_grid': param_grid,
    }

    data = pd.read_csv(source_data,
                       index_col=0)

    estimator, best_params, perf_scores = tuning(data, dataset_params, model_params)

    perf_scores.insert(0, 'source', source_data)
    output_dir = f'outputs/tuning/{league_name}/'
    ensure_folder(output_dir)

    output_path = f'{output_dir}{test_name}_fine_tuning_{get_timestamp()}.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        perf_scores.to_excel(writer, sheet_name="Tuning", index=False)
        best_params.to_excel(writer, sheet_name="Best Params", index=True)
