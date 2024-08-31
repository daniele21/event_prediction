import io

import pandas as pd
from flask import current_app as app, send_file, render_template_string
from flask import make_response, request

from api.templates.table_template import HTML_TEMPLATE
from core.ingestion.update_league_data import get_most_recent_data
from core.logger import logger
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_margin_matches
from core.strategies.net_strategy import positive_net_strategy
from core.utils import load_pickle, load_json
from scripts.e2e import net_prediction_model
from scripts.model_inference import classification_model_inference
from scripts.update_data import get_next_match_day_data


@app.route('/api/go_live', methods=['POST'])
def go_live():
    payload = request.json
    source_folder = payload['source_folder']

    dataset_path = f'{source_folder}/dataset_config.json'
    dataset_params = load_json(dataset_path)
    logger.info(dataset_params)

    class_model_path = f'{source_folder}/class_model.pkl'
    net_model_path = f'{source_folder}/net_prediction_model.pkl'
    class_model = load_pickle(class_model_path)
    net_model = load_pickle(net_model_path)

    try:
        next_match_data = get_next_match_day_data(dataset_params)
        bookmakers = next_match_data['bookmaker']
        next_match_data = next_match_data.drop('bookmaker', axis=1)
        prediction, probs = classification_model_inference(next_match_data,
                                                           dataset_params,
                                                           class_model,
                                                           go_live=True)

        # Simulation
        simulation_data = enrich_data_for_simulation(next_match_data, probs)
        simulation_data = simulation_data.merge(bookmakers, left_index=True, right_index=True)
        plays = extract_margin_matches(simulation_data, bookmakers=True)

        df, _ = positive_net_strategy(plays, net_model, inference=True)
        df = df.drop(['win', 'gain', 'net',
                      'net_rate', 'index',
                      'result_1X2'], axis=1).sort_index()
        df = df[df['bet_decision'] == 1].drop('bet_decision', axis=1)
        html_table = df.to_html(classes='data', header=True, index=False)

        # Render the HTML template with the table data
        return render_template_string(HTML_TEMPLATE, table=html_table)

    except Exception as e:
        tb = e.__traceback__
        msg = f'{str(e.with_traceback(tb))}'
        response = make_response(msg, 500)

    return response
