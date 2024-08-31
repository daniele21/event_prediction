from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import config.league as LEAGUE
from config.constants import MATCH_RESULT_V1
from core.dataset.dataset_generation import generate_datasets
from core.ingestion.update_league_data import update_data_league
from core.utils import get_most_recent_data
import numpy as np
import pandas as pd
from models.training_process import training_process
import matplotlib.pyplot as plt


def season_analysis(data):
    last_n_seasons = np.arange(1, len(data['season'].unique()))
    result_dict = {}

    rf_params = {'n_estimators': 100,
                 'max_depth': 10,
                 'random_state': 2024,
                 'n_jobs': -1}
    estimator = RandomForestClassifier

    for last_seasons in tqdm(last_n_seasons):
        dataset_params = {'drop_last': 5,
                          'drop_first': 5,
                          'last_n_seasons': last_seasons,
                          'target_match_days': np.arange(9, 33),
                          'test_match_day': 2,
                          'preprocessing_version': MATCH_RESULT_V1,
                          }

        datasets = generate_datasets(data, dataset_params)
        features = datasets[9]['train']['x'].columns
        model, model_scores, probabilities = training_process(datasets, estimator, rf_params, inference=False)
        result_dict[last_seasons] = {'model': model,
                                     'scores': model_scores.rename(columns={'precision': f'precision_{last_seasons}',
                                                                            'recall': f'recall_{last_seasons}',
                                                                            'f1': f'f1_{last_seasons}'
                                                                            }),
                                     'avg_precision': model_scores['precision'].mean(),
                                     'probabilities': probabilities,
                                     'importance': pd.DataFrame(model.feature_importances_.reshape(1, -1),
                                                                columns=features).T.rename(columns={0: last_seasons})
                                     }
    feature_importance = pd.DataFrame(index=features)
    performance = pd.DataFrame(index=model_scores['target_day'])
    for key, value in result_dict.items():
        feature_importance = feature_importance.merge(result_dict[key]['importance'], left_index=True, right_index=True)
        cols = [f'precision_{key}', f'recall_{key}', f'f1_{key}']
        performance = performance.merge(result_dict[key]['scores'].set_index('target_day')[cols], left_index=True, right_index=True)

    precision_cols = [x for x in performance.columns if 'precision' in x]
    recall_cols = [x for x in performance.columns if 'recall' in x]
    f1_cols = [x for x in performance.columns if 'f1' in x]
    performance[precision_cols].mean(axis=0).plot(label='precision')
    performance[recall_cols].mean(axis=0).plot(label='recall')
    performance[f1_cols].mean(axis=0).plot(label='f1')
    plt.title(data['league'].iloc[0])
    plt.legend()
    plt.grid()
    plt.show()

    return
    # plt.plot(model_scores['target_day'], model_scores['precision'], marker='o')
    # plt.axhline(model_scores['precision'].mean(), c='r')
    # plt.axhline(model_scores['precision'].median(), c='orange')
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    league_name = LEAGUE.SERIE_A
    windows=[1,3,5]

    league_dir = f'resources/{league_name}/'
    source_path = get_most_recent_data(league_dir, league_name,
                                       windows=windows)

    if source_path:
        data = pd.read_csv(source_path,
                           index_col=0)
    else:
        params = {'league_name': league_name,
                  'windows': windows,
                  'league_dir': f"resources/",
                  'update': True}
        data = update_data_league(params)

    season_analysis(data)
