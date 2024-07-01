import unittest
import config.league as LEAGUE
from core.ingestion.load_data import extract_data


class DataTests(unittest.TestCase):
    def test_extract_season_data(self):
        league_name = LEAGUE.SERIE_A
        data = extract_data(league_name)

        self.assertIsNotNone(data)

if __name__ == '__main__':
    unittest.main()
