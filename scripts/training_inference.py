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


def train_and_inference(data, dataset_params, estimator, best_params):
    datasets = generate_datasets(data, dataset_params)

    final_plays = {}
    net = {}
    total_gains = {}
    for metric, params in best_params.items():
        model, model_scores, probabilities = training_process(datasets, estimator, params, inference=True)

        # Simulation
        simulation_data = enrich_data_for_simulation(data, probabilities)
        plays = extract_margin_matches(simulation_data)
        final_plays[metric] = plays

    # Save
    for k, _ in final_plays.items():
        for name, param in best_params[k].items():
            final_plays[k].insert(0, name, param)

        for key, value in dataset_params.items():
            final_plays[k].insert(0, key, str(value))

    return final_plays, net, total_gains


if __name__ == '__main__':
    league_name = LEAGUE.SERIE_A
    npm = 5
    test_name = 'target_2324'

    league_dir = f'resources/{league_name}/'
    source_data = get_most_recent_data(league_dir, league_name, n_prev_match=npm)
    dataset_params = {'drop_last': 5,
                      'drop_first': 5,
                      'last_n_seasons': 13,
                      'target_match_days': np.arange(7, 31),
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
        'recall': {'num_leaves': 5, 'max_depth': 5, 'feature_fraction': 0.8,
                   'learning_rate': 0.01, 'n_estimators': 300,
                   'class_weight': None,
                   'deterministic': True, 'seed': 2024},
        'precision': {'num_leaves': 15, 'max_depth': 5, 'feature_fraction': 1,
                      'learning_rate': 0.01, 'n_estimators': 300,
                      'class_weight': 'balanced',
                      'deterministic': True, 'seed': 2024},
        'f1': {'num_leaves': 15, 'max_depth': 5, 'feature_fraction': 0.8,
               'learning_rate': 0.01, 'n_estimators': 300,
               'class_weight': 'balanced',
               'deterministic': True, 'seed': 2024},
        'log_loss': {'num_leaves': 15, 'max_depth': 10, 'feature_fraction': 1,
                     'learning_rate': 0.5, 'n_estimators': 300,
                     'class_weight': None,
                     'deterministic': True, 'seed': 2024},

    }

    estimator = lgbm.LGBMClassifier
    # estimator = RandomForestClassifier

    data = pd.read_csv(source_data,
                       index_col=0)

    final_plays, net, total_gains = train_and_inference(data, dataset_params, estimator, best_params)

    output_dir = f'outputs/simulations/{league_name}/'
    ensure_folder(output_dir)

    output_path = f'{output_dir}{test_name}_simulation_model_{get_timestamp()}.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for k, _ in final_plays.items():
            final_plays[k].insert(0, 'source', source_data)
            final_plays[k].to_excel(writer, sheet_name=k, index=False)
