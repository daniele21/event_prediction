import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm

from config.constants import MATCH_RESULT_V1
import config.league as LEAGUE
from core.dataset.dataset_generation import generate_datasets
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_margin_matches
from core.utils import get_timestamp, ensure_folder, get_most_recent_data
from models.training_framework import hyperparameter_tuning
from models.training_process import training_process
from scripts.training.validation import check_calibration


def train_and_test(data, dataset_params, estimator, params, inference=False):
    datasets = generate_datasets(data, dataset_params)

    model, model_scores, probabilities = training_process(datasets, estimator, params, inference=inference)

    # Simulation
    simulation_data = enrich_data_for_simulation(data, probabilities)
    plays = extract_margin_matches(simulation_data)

    # check_calibration(plays)

    # Add params
    for name, param in params.items():
        plays.insert(0, name, param)

    for key, value in dataset_params.items():
        plays.insert(0, key, str(value))

    return plays


if __name__ == '__main__':
    league_name = LEAGUE.SERIE_A
    windows = [1, 3, 5]
    test_name = 'target_2324'
    # import sys
    # print('------------')
    # print(sys.executable)
    league_dir = f'resources/{league_name}/'
    source_data = get_most_recent_data(league_dir, league_name, windows)
    dataset_params = {'drop_last': 5,
                      'drop_first': 5,
                      'last_n_seasons': 13,
                      'target_match_days': np.arange(8, 30),
                      #'target_match_days': [7, 10, 13],
                      'test_match_day': 2,
                      'preprocessing_version': MATCH_RESULT_V1,
                      }

    # RF
    # best_params = {
    #     'recall': {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 2,
    #                'min_samples_leaf': 2, 'criterion': 'entropy',
    #                'class_weight': 'balanced', 'random_state': 2024},
    #     'precision': {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 2,
    #                   'min_samples_leaf': 4, 'criterion': 'gini',
    #                   'class_weight': 'balanced', 'random_state': 2024},
    #     'f1': {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 5,
    #            'min_samples_leaf': 2, 'criterion': 'entropy',
    #            'class_weight': 'balanced', 'random_state': 2024},
    #     'log_loss': {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 2,
    #                  'min_samples_leaf': 1, 'criterion': 'gini',
    #                  'class_weight': None, 'random_state': 2024}
    # }

    # LGBM
    best_params = {
        'log_loss': {'num_leaves': 5, 'max_depth': 5,
                     'learning_rate': 0.1, 'n_estimators': 100,
                     'early_stopping_round': 10, 'class_weight': None,
                     'deterministic': True, 'seed': 2024},

        'precision': {'num_leaves': 15, 'max_depth': 10,
                      'learning_rate': 0.1, 'n_estimators': 100,
                      'early_stopping_round': 10, 'class_weight': None,
                      'deterministic': True, 'seed': 2024},

    }

    estimator = lgbm.LGBMClassifier
    # estimator = RandomForestClassifier

    data = pd.read_csv(source_data,
                       index_col=0)

    test_plays = train_and_test(data, dataset_params, estimator, best_params['log_loss'],
                                inference=False)
    target_plays = train_and_test(data, dataset_params, estimator, best_params['log_loss'],
                                  inference=True)

    output_dir = f'outputs/training_test/{league_name}/'
    ensure_folder(output_dir)

    output_path = f'{output_dir}{test_name}_test_model_{get_timestamp()}.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        test_plays.insert(0, 'source', source_data)
        test_plays.to_excel(writer, sheet_name='test', index=False)
        target_plays.insert(0, 'source', source_data)
        target_plays.to_excel(writer, sheet_name='target', index=False)

    #
    # final_plays_1 = train_and_test(data, dataset_params, estimator, best_params['log_loss'],
    #                              inference=False)
    # final_plays_2 = train_and_test(data, dataset_params, estimator, best_params['log_loss'],
    #                              inference=True)
    # final_plays_3 = train_and_test(data, dataset_params, estimator, best_params['precision'],
    #                                inference=False)
    # final_plays_4 = train_and_test(data, dataset_params, estimator, best_params['precision'],
    #                                inference=True)
    #
    # check_calibration(final_plays_1, prob_col='prob', title='log_loss + test + prob')
    # check_calibration(final_plays_1, prob_col='bet_prob', title='log_loss + test + bet_prob')
    # check_calibration(final_plays_2, prob_col='prob', title='log_loss + target + prob')
    # check_calibration(final_plays_2, prob_col='bet_prob', title='log_loss + target + bet_prob')
    # check_calibration(final_plays_3, prob_col='prob', title='precision + test + prob')
    # check_calibration(final_plays_3, prob_col='bet_prob', title='precision + test + bet_prob')
    # check_calibration(final_plays_4, prob_col='prob', title='precision + target + prob')
    # check_calibration(final_plays_4, prob_col='bet_prob', title='precision + target + bet_prob')
