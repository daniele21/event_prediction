import json
import os
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgbm

def load_json(filepath):
    with open(filepath, 'rb') as f:
        content = json.load(f)
        f.close()

    return content


def save_json(content, filepath):
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=4)
        f.close()


def load_pickle(input_filename):
    with open(input_filename, 'rb') as filename:
        loaded_object = pickle.load(filename)
    return loaded_object


def save_pickle(input_object, output_filename):
    with open(output_filename, 'wb') as filename:
        pickle.dump(input_object, filename)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
        return True

    return False

def get_estimator(estimator):
    if isinstance(estimator, str):
        if str("RandomForestClassifier").lower() == estimator.lower():
            return RandomForestClassifier
        elif str("RandomForestRegressor").lower() == estimator.lower():
            return RandomForestRegressor
        elif str("LGBMClassifier").lower() == estimator.lower():
            return lgbm.LGBMClassifier
        elif str("LGBMRegressor").lower() == estimator.lower():
            return lgbm.LGBMRegressor
        else:
            raise AttributeError(f'Estimator not found: {estimator}')
    else:
        return estimator

def get_most_recent_data_path(league_dir, league_name, windows):
    if league_dir is None or not os.path.exists(league_dir):
        return None

    else:
        filename_pattern = f'{league_name}_{"-".join([str(x) for x in windows])}_'
        items = [item for item in os.listdir(league_dir) if filename_pattern in item]
        sorted_items = sorted(items)
        filename = sorted_items[-1] if len(sorted_items) > 0 else None

        if filename:
            league_path = f'{league_dir}{filename}'
        else:
            league_path = None

        return league_path
