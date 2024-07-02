import unittest
import config.league as LEAGUE
from core.ingestion.load_data import extract_data
from core.ingestion.update_league_data import update_data_league


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


if __name__ == '__main__':
    unittest.main()
