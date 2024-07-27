from datetime import datetime
import os


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
        return True

    return False


def get_most_recent_data(league_dir, league_name, n_prev_match):
    if league_dir is None or not os.path.exists(league_dir):
        return None

    else:
        items = os.listdir(league_dir)
        sorted_items = sorted(items)
        filename = sorted_items[-1]

        league_path = f'{league_dir}{filename}'
        return league_path
