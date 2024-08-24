from datetime import datetime
import os



def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
        return True

    return False


def get_most_recent_data(league_dir, league_name, windows):
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
