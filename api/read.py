from flask import make_response, jsonify, request, render_template

from scripts.data.data_extraction import teams_extraction
from scripts.utils.checker import check_league

import scripts.constants.league as LEAGUE


from flask import current_app as app


@app.route('/api/v1/read/leagues', methods=['GET'])
def get_league_names():
    return make_response(jsonify({'league_name': LEAGUE.LEAGUE_NAMES}))


@app.route('/api/v1/read/<league_name>/team', methods=['GET'])
def get_teams(league_name):
    """

    Args:
        league_name: str

    Returns:
        response: dict { 'league_name': str,
                         'teams': list }
    """

    outcome, msg = check_league(league_name)
    if not outcome:
        response = make_response(msg, 404)
    else:
        # teams = sorted(LEAGUE.TEAMS_LEAGUE[str(league_name).lower()])
        # response = make_response(jsonify({'league_name': league_name,
        #                                   'teams': teams}))

        teams = teams_extraction(league_name)
        teams.sort()

        response_dict = {'league_name': league_name,
                    'teams': teams}

        response = make_response(response_dict, 200)

    return response

@app.route('/api/v1/read/teams', methods=['GET'])
def get_teams_from_league():
    """

    Requested Args:
        league_name: str

    Returns:
        response: dict { 'league_name': str,
                         'teams': list }
    """

    args = request.args
    league_name = args['league']

    outcome, msg = check_league(league_name)
    if not outcome:
        response = make_response(msg, 404)
    else:
        # teams = sorted(LEAGUE.TEAMS_LEAGUE[str(league_name).lower()])
        # response = make_response(jsonify({'league_name': league_name,
        #                                   'teams': teams}))

        teams = teams_extraction(league_name)
        teams.sort()

        response_dict = {'league_name': league_name,
                         'teams': teams}

        response = make_response(response_dict, 200)

    return response

@app.route('/api/v1/read/matches', methods=['GET'])
def get_round():

    """
    Requested Args:
        - league_name: str

    """

    args = request.args

    league_name = args['league_name']

    outcome, msg = check_league(league_name)
    if not outcome:
        response = make_response(msg, 404)
    else:
        pass
        # compute next round matches