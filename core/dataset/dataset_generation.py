import numpy as np

from core.dataset.preprocessing_functions import dataset_preprocessing
from scripts.training.data_split import split_data


def generate_datasets(data, dataset_params):
    preprocessing_fn = dataset_preprocessing(dataset_params['preprocessing_version'])
    x, y = preprocessing_fn(data)

    if isinstance(dataset_params['target_match_days'], dict):
        target_match_days = np.arange(dataset_params['target_match_days']['start'],
                                      dataset_params['target_match_days']['end'])
    else:
        target_match_days = dataset_params['target_match_days']

    datasets = {}
    for target_match_day in target_match_days:
        dataset = split_data(x, y,
                             target_match_day,
                             dataset_params['test_match_day'],
                             dataset_params['last_n_seasons'],
                             dataset_params['drop_last_seasons'],
                             dataset_params['drop_first_match_days'],
                             dataset_params['drop_last_match_days'])
        datasets[target_match_day] = dataset

    return datasets
