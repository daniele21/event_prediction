# -*- coding: utf-8 -*-

from config.league import SERIE_A, SERIE_A_PATH, PREMIER, PREMIER_PATH, JUPILIER, JUPILIER_PATH, LIGUE_1, \
    LIGUE_1_PATH, LEAGUE_NAMES, EREDIVISIE, EREDIVISIE_PATH, LIGA, LIGA_PATH, LIGA_2, LIGA_2_PATH


def get_league_csv_paths(league_name):
    paths = []

    if league_name == SERIE_A:
        paths = SERIE_A_PATH

    elif league_name == PREMIER:
        paths = PREMIER_PATH

    elif league_name == JUPILIER:
        paths = JUPILIER_PATH

    elif league_name == LIGUE_1:
        paths = LIGUE_1_PATH

    elif league_name == EREDIVISIE:
        paths = EREDIVISIE_PATH

    elif league_name == LIGA:
        paths = LIGA_PATH

    elif league_name == LIGA_2:
        paths = LIGA_2_PATH

    return paths
