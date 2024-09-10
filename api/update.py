import json

from flask import current_app as app
from flask import make_response, jsonify, request, render_template
import json
from core.ingestion.update_league_data import update_data_league
from core.logger import logger


@app.route('/api/update', methods=['POST'])
def update_league():

    payload = request.json

    league_name = payload['league_name']
    windows = payload['windows']
    league_dir = payload.get('league_dir')

    params = {'league_name': league_name,
              'windows': windows,
              'league_dir': league_dir if league_dir is not None else "resources/",
              'update': True}

    try:
        data = update_data_league(params)
        msg = f"Updated league_name: {league_name}"
        response = make_response(msg, 200)
    except Exception as e:
        tb = e.__traceback__
        msg = f'{str(e.with_traceback(tb))}'
        response = make_response(msg, 500)

    return response


