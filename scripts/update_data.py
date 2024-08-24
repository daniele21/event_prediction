import config.league as LEAGUE
from core.ingestion.update_league_data import update_data_league


def create_update_league_data(params):
    data = update_data_league(params)

    return data


if __name__ == '__main__':
    league_name = LEAGUE.SERIE_A
    npm = 5
    params = {'league_name': league_name,
              'windows': [1, 3, 5],
              'league_dir': f"resources/",
              'update': True}

    data = create_update_league_data(params)
