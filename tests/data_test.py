import unittest
import config.league as LEAGUE
from core.ingestion.load_data import extract_data
from core.ingestion.update_league_data import update_data_league
from scripts.training.data_split import split_data
import pandas as pd


class DataTests(unittest.TestCase):
    def test_extract_season_data(self):
        league_name = LEAGUE.SERIE_A
        data = extract_data(league_name, n_prev_match=5)

        self.assertIsNotNone(data)

    def test_update_data_league(self):
        league_name = LEAGUE.SERIE_A
        npm = 5
        params = {'league_name': league_name,
                  'n_prev_match': npm,
                  'league_dir': f"resources/",
                  'update': True}
        data = update_data_league(params)

        self.assertIsNotNone(data)

    def test_split_data(self):
        data = pd.read_csv('resources/serie_a/serie_a_npm=5.csv', index_col=0)
        target = ['result_1X2']
        drop_cols = ['home_goals', 'away_goals']
        data = data.drop(drop_cols, axis=1)
        x = data.drop(target, axis=1)
        y = data[target]

        drop_first = 5
        drop_last = 5
        last_n_seasons = 5
        target_match_day = 18
        test_match_day = 2
        dataset = split_data(x, y, target_match_day, test_match_day,
                             last_n_seasons, drop_first, drop_last)
        x_train, y_train = dataset['train']['x'], dataset['train']['y']
        x_test, y_test = dataset['test']['x'], dataset['test']['y']
        x_target, y_target = dataset['target']['x'], dataset['target']['y']

        self.assertIsNotNone(dataset)

        self.assertEqual(len(x_test[['season', 'match_day']].drop_duplicates()), test_match_day)
        self.assertEqual(len(x_target[['season', 'match_day']].drop_duplicates()), 1)
        self.assertEqual(x_target['match_day'].drop_duplicates().values.item(), target_match_day)

    def test_performance_split(self):
        data = pd.read_csv('resources/serie_a/serie_a_npm=5.csv', index_col=0)
        target = ['result_1X2']
        drop_cols = ['home_goals', 'away_goals']
        data = data.drop(drop_cols, axis=1)
        x = data.drop(target, axis=1)
        y = data[target]

        drop_first = 5
        drop_last = 5
        last_n_seasons = 5
        target_match_day_list = [9, 14, 18, 22, 27]
        test_match_day = 2

        datasets = {}

        for target_match_day in target_match_day_list:
            dataset = split_data(x, y, target_match_day, test_match_day,
                                 last_n_seasons, drop_first, drop_last)

            datasets[target_match_day] = dataset

        return

if __name__ == '__main__':
    unittest.main()
