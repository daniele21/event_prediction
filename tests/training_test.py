import unittest

import pandas as pd

from config.constants import MATCH_RESULT_V1
from core.dataset.dataset_generation import generate_datasets
from core.grid_search import grid_search
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_positive_margin_matches
from models.inference import model_inference
from models.training_framework import hyperparameter_tuning
from models.training_process import training_process
from scripts.training.data_split import split_data
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class TrainingTest(unittest.TestCase):

    def test_hyperparameter_tuning(self):
        drop_first = 5
        drop_last = 5
        last_n_seasons = 5
        target_match_days = [9, 13, 17, 21, 24, 27, 32]
        test_match_day = 2

        scoring = 'recall'
        estimator = RandomForestClassifier
        param_grid = {'n_estimators': [100],
                      'max_depth': [3, 5, 7, 9, 13],
                      'random_state': [2024],
                      }

        dataset_params = {'drop_last': drop_last,
                          'drop_first': drop_first,
                          'last_n_seasons': last_n_seasons,
                          'target_match_days': target_match_days,
                          'test_match_day': test_match_day,
                          'scoring': scoring,
                          'preprocessing_version': MATCH_RESULT_V1,
                          }
        model_params = {'estimator': estimator,
                        'param_grid': param_grid,
                        'scoring': scoring}

        params = {'dataset': dataset_params,
                  'model': model_params}

        data = pd.read_csv('resources/serie_a/serie_a_npm=5.csv',
                           index_col=0)

        datasets = generate_datasets(data, dataset_params)
        estimator, best_params = hyperparameter_tuning(datasets, model_params)

        self.assertIsNotNone(estimator)
        self.assertIsNotNone(best_params)

    def test_training_best_params(self):
        drop_first = 5
        drop_last = 5
        last_n_seasons = 5
        target_match_days = [9, 13, 17, 21, 24, 27, 32]
        test_match_day = 2

        scoring = 'recall'
        estimator = RandomForestClassifier
        param_grid = {'n_estimators': [100],
                      'max_depth': [3, 5, 7, 9, 13],
                      'random_state': [2024],
                      }

        dataset_params = {'drop_last': drop_last,
                          'drop_first': drop_first,
                          'last_n_seasons': last_n_seasons,
                          'target_match_days': target_match_days,
                          'test_match_day': test_match_day,
                          'scoring': scoring,
                          'preprocessing_version': MATCH_RESULT_V1,
                          }
        model_params = {'estimator': estimator,
                        'param_grid': param_grid,
                        'scoring': scoring}

        data = pd.read_csv('resources/serie_a/serie_a_npm=5.csv',
                           index_col=0)

        datasets = generate_datasets(data, dataset_params)
        estimator, best_params = hyperparameter_tuning(datasets, model_params)

        model, model_scores, probabilities = training_process(datasets, estimator, best_params)

        self.assertIsNotNone(model)
        self.assertIsNotNone(model_scores)
        self.assertIsNotNone(probabilities)

    def test_simulation(self):
        drop_first = 5
        drop_last = 5
        last_n_seasons = 5
        #target_match_days = [9, 13, 17, 21, 24, 27, 32]
        target_match_days = np.arange(6, 33)
        test_match_day = 2

        scoring = 'recall'
        estimator = RandomForestClassifier
        param_grid = {'n_estimators': [100],
                      'max_depth': [3, 5, 7, 9, 13],
                      'random_state': [2024],
                      }

        dataset_params = {'drop_last': drop_last,
                          'drop_first': drop_first,
                          'last_n_seasons': last_n_seasons,
                          'target_match_days': target_match_days,
                          'test_match_day': test_match_day,
                          'scoring': scoring,
                          'preprocessing_version': MATCH_RESULT_V1,
                          }

        data = pd.read_csv('resources/serie_a/serie_a_npm=5.csv',
                           index_col=0)

        datasets = generate_datasets(data, dataset_params)

        best_params = {'max_depth': 7, 'n_estimators': 100, 'random_state': 2024}
        model, model_scores, probabilities = training_process(datasets, estimator, best_params)

        # Simulation
        simulation_data = enrich_data_for_simulation(data, probabilities)
        plays = extract_positive_margin_matches(simulation_data)
        gain_by_match_day = plays.groupby('match_day')['gain'].sum()
        total_gain = plays['gain'].sum()

        self.assertIsNotNone(simulation_data)

    def test_grid_search(self):
        drop_first = 5
        drop_last = 5
        last_n_seasons = 5
        target_match_days = [9, 13, 17, 21, 24, 27, 32]
        test_match_day = 2
        scoring = 'recall'

        data = pd.read_csv('resources/serie_a/serie_a_npm=5.csv',
                           index_col=0)
        target = ['result_1X2']
        drop_target_cols = ['home_goals', 'away_goals',
                            'home_points', 'away_points']
        data = data.drop(drop_target_cols, axis=1)
        drop_cols = ['league', 'AwayTeam', 'HomeTeam', 'Date',
                     'match_n', 'bet_1', 'bet_X', 'bet_2']
        data = data.drop(drop_cols, axis=1)
        x = data.drop(target, axis=1).fillna(0)
        y = data[target]

        le = LabelEncoder()
        encoded_y = y.copy(deep=True)
        encoded_y[target[0]] = le.fit_transform(encoded_y)

        datasets = {}
        for target_match_day in tqdm(target_match_days):
            dataset = split_data(x, encoded_y, target_match_day, test_match_day,
                                 last_n_seasons, drop_first, drop_last)
            datasets[target_match_day] = dataset

        estimator = RandomForestClassifier

        param_grid = {'n_estimators': [100],
                      'max_depth': [3, 5, 7, 9, 13],
                      'random_state': [2024],
                      }

        performance_scores = grid_search(estimator, datasets, param_grid)
        best_params = performance_scores.drop(['target_day', 'recall',
                                               'precision', 'f1'], axis=1) \
            .drop_duplicates() \
            .sort_values(by=f'avg_{scoring}', ascending=False) \
            .drop(['estimator', 'avg_recall',
                   'avg_precision', 'avg_f1'], axis=1) \
            .iloc[0] \
            .to_dict()

        self.assertIsNotNone(best_params, performance_scores)


if __name__ == '__main__':
    unittest.main()
