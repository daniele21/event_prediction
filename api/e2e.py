import io
import json

import pandas as pd
from flask import current_app as app, render_template_string, send_file
from flask import make_response, jsonify, request, render_template
import json

from api.templates.table_template import HTML_TEMPLATE
from core.ingestion.update_league_data import update_data_league, get_most_recent_data
from core.logger import logger
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_margin_matches
from core.strategies.net_strategy import positive_net_strategy
from core.utils import get_timestamp, ensure_folder, load_json
from scripts.e2e import ht_section, training_and_test_section, net_prediction_model
from scripts.model_inference import classification_model_inference


@app.route('/api/training_test_e2e', methods=['POST'])
def training_test_e2e_process():
    payload = request.json
    exp_name = payload['exp_name']
    dataset_params = payload['dataset']
    training_params = payload['training']
    strategy_params = payload['strategy']
    source_folder = payload.get('source_folder')

    e2e_folder = f'outputs/e2e/{dataset_params["league_name"]}/{exp_name}_{get_timestamp()}'
    source_folder = e2e_folder if source_folder is None else source_folder

    ensure_folder(source_folder)

    source_data = get_most_recent_data(dataset_params)

    try:
        best_params = ht_section(source_data, dataset_params, training_params, save_folder=source_folder)
        best_params_str = str(best_params).replace("'", '"')
        msg = f"Tuning results: \n\nModel: {training_params['estimator']}\n\n{best_params_str}\nOptimized score: {training_params['scoring']}\n\n"
        msg += f"Saved at\n{source_folder}"

        class_model, bet_plays, fig = training_and_test_section(source_data,
                                                                dataset_params,
                                                                training_params,
                                                                best_params,
                                                                save_folder=source_folder)

        bet_plays_path = f'{source_folder}/bet_plays.csv'
        bet_plays = pd.read_csv(bet_plays_path, index_col=0)
        net_model, x_test = net_prediction_model(bet_plays, strategy_params, save_folder=source_folder)

        _ = positive_net_strategy(x_test, net_model, info_data='test set', save_folder=source_folder)

        prediction, probs = classification_model_inference(source_data, dataset_params, class_model)

        # Simulation
        simulation_data = enrich_data_for_simulation(source_data, probs)
        plays = extract_margin_matches(simulation_data)

        fig = positive_net_strategy(plays, net_model, info_data='target set', save_folder=source_folder)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        response = send_file(img, mimetype='image/png')
    except Exception as e:
        tb = e.__traceback__
        msg = f'{str(e.with_traceback(tb))}'
        response = make_response(msg, 500)

    return response


@app.route('/api/training', methods=['POST'])
def training():
    payload = request.json
    exp_name = payload.get('exp_name')
    dataset_params = payload.get('dataset')
    training_params = {'estimator': payload['estimator']}
    params = payload.get('params')
    source_folder = payload.get('source_folder')

    if dataset_params is None:
        dataset_path = f'{source_folder}/dataset_config.json'
        dataset_params = load_json(dataset_path)
        logger.info(dataset_params)

    if params is None:
        params_path = f'{source_folder}/tuning_best_params.json'
        params = load_json(params_path)
        logger.info(params)

    e2e_folder = f'outputs/e2e/{dataset_params["league_name"]}/{exp_name}_{get_timestamp()}'
    source_folder = e2e_folder if source_folder is None else source_folder

    ensure_folder(source_folder)

    source_data = get_most_recent_data(dataset_params)

    try:
        class_model, bet_plays, fig = training_and_test_section(source_data,
                                                                dataset_params,
                                                                training_params,
                                                                params,
                                                                save_folder=source_folder)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        response = send_file(img, mimetype='image/png')

        # # Convert the DataFrame to an HTML table
        # html_table = bet_plays.to_html(classes='data', header=True, index=False)
        #
        # # Render the HTML template with the table data
        # response = render_template_string(HTML_TEMPLATE, table=html_table)

    except Exception as e:
        response = make_response(str(e), 500)

    return response
