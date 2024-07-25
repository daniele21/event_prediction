from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from core.dataset.preprocessing_functions import dataset_preprocessing
from scripts.training.data_split import split_data


def generate_datasets(data, dataset_params):
    preprocessing_fn = dataset_preprocessing(dataset_params['preprocessing_version'])
    x, y = preprocessing_fn(data)

    datasets = {}
    for target_match_day in dataset_params['target_match_days']:
        dataset = split_data(x, y,
                             target_match_day,
                             dataset_params['test_match_day'],
                             dataset_params['last_n_seasons'],
                             dataset_params['drop_first'],
                             dataset_params['drop_last'])
        datasets[target_match_day] = dataset

    return datasets
