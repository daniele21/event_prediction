import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.constants import MATCH_RESULT_V1
import config.league as LEAGUE
from core.dataset.dataset_generation import generate_datasets
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_positive_margin_matches
from core.utils import get_timestamp, ensure_folder, get_most_recent_data
from models.training_framework import hyperparameter_tuning
from models.training_process import training_process


def train_and_inference(data, dataset_params, estimator, best_params):
    datasets = generate_datasets(data, dataset_params)

    final_plays = {}
    gains = {}
    total_gains = {}
    for metric, params in best_params.items():
        model, model_scores, probabilities = training_process(datasets, estimator, params)

        # Simulation
        simulation_data = enrich_data_for_simulation(data, probabilities)
        plays = extract_positive_margin_matches(simulation_data)
        gain_by_match_day = plays.groupby('match_day')['gain'].sum()
        total_gain = plays['gain'].sum()
        final_plays[metric] = plays
        gains[metric] = gain_by_match_day
        total_gains[metric] = total_gain

    for k, _ in final_plays.items():
        for name, param in best_params[k].items():
            final_plays[k].insert(0, name, param)

        for key, value in dataset_params.items():
            final_plays[k].insert(0, key, str(value))

    return final_plays, gains, total_gains


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

    best_params = {
        'recall': {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 10,
                   'min_samples_leaf': 2, 'bootstrap': False, 'random_state': 2024},
        'precision': {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 2,
                     'min_samples_leaf': 1, 'bootstrap': False, 'random_state': 2024},
        'f1': {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 5,
                 'min_samples_leaf': 1, 'bootstrap': False, 'random_state': 2024}
    }

    estimator = RandomForestClassifier

    data = pd.read_csv(source_data,
                       index_col=0)

    final_plays, gains, total_gains = train_and_inference(data, dataset_params, estimator, best_params)

    output_dir = f'outputs/simulations/{league_name}/'
    ensure_folder(output_dir)

    for k, _ in final_plays.items():
        final_plays[k].insert(0, 'source', source_data)
        final_plays[k].to_csv(f'{output_dir}{test_name}_simulation_model_optimizing_{k}_{get_timestamp()}.csv')
