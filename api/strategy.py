import io

import pandas as pd
from flask import current_app as app, send_file
from flask import make_response, request

from core.ingestion.update_league_data import get_most_recent_data
from core.logger import logger
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_margin_matches
from core.strategies.net_strategy import positive_net_strategy, high_positive_net_strategy
from core.utils import load_pickle, load_json
from scripts.new_e2e import net_prediction_model
from scripts.model_inference import classification_model_inference


@app.route('/api/strategy/net_prediction_model', methods=['POST'])
def net_prediction():
    payload = request.json
    strategy_params = payload['strategy']
    source_folder = payload['source_folder']

    try:
        bet_plays_path = f'{source_folder}/test_bet_plays.csv'
        bet_plays = pd.read_csv(bet_plays_path, index_col=0)
        net_model_dict, net_scaler_dict, x_test = net_prediction_model(bet_plays,
                                                 strategy_params,
                                                 save_folder=source_folder)

        _, fig = positive_net_strategy(x_test,
                                       net_model_dict,
                                       net_scaler_dict,
                                       info_data='test set',
                                       save_folder=source_folder)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        response = send_file(img, mimetype='image/png')
    except Exception as e:
        tb = e.__traceback__
        msg = f'{str(e.with_traceback(tb))}'
        logger.error(msg)
        response = make_response(msg, 500)

    return response


@app.route('/api/strategy/positive_net', methods=['POST'])
def positive_net_strategy_api():
    payload = request.json
    dataset_params = payload.get('dataset')
    source_folder = payload['source_folder']
    target_match_day = payload.get("target_match_days")

    if dataset_params is None:
        dataset_path = f'{source_folder}/dataset_config.json'
        dataset_params = load_json(dataset_path)
        logger.info(dataset_params)

    if target_match_day:
        dataset_params["target_match_days"] = target_match_day

    class_model_path = f'{source_folder}/class_model.pkl'
    net_model_path = f'{source_folder}/net_prediction_model.pkl'
    net_scaler_path = f'{source_folder}/net_scaler.pkl'
    class_model = load_pickle(class_model_path)
    net_model_dict = load_pickle(net_model_path)
    net_scaler_dict = load_pickle(net_scaler_path)

    data = get_most_recent_data(dataset_params)

    try:
        prediction, probs = classification_model_inference(data, dataset_params, class_model)

        # Simulation
        simulation_data = enrich_data_for_simulation(data, probs[['1', 'X', '2']])
        plays = extract_margin_matches(simulation_data)
        plays['target_day'] = plays['match_day']

        _, fig = positive_net_strategy(plays,
                                       net_model_dict,
                                       net_scaler_dict,
                                       info_data='target set',
                                       save_folder=source_folder)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        response = send_file(img, mimetype='image/png')

    except Exception as e:
        tb = e.__traceback__
        msg = f'{str(e.with_traceback(tb))}'
        logger.error(msg)
        response = make_response(msg, 500)

    return response


@app.route('/api/strategy/high_positive_net', methods=['POST'])
def high_positive_net_strategy_api():
    payload = request.json
    dataset_params = payload.get('dataset')
    source_folder = payload['source_folder']

    if dataset_params is None:
        dataset_path = f'{source_folder}/dataset_config.json'
        dataset_params = load_json(dataset_path)
        logger.info(dataset_params)

    class_model_path = f'{source_folder}/class_model.pkl'
    net_model_path = f'{source_folder}/net_prediction_model.pkl'
    class_model = load_pickle(class_model_path)
    net_model = load_pickle(net_model_path)

    data = get_most_recent_data(dataset_params)

    try:
        prediction, probs = classification_model_inference(data, dataset_params, class_model)

        # Simulation
        simulation_data = enrich_data_for_simulation(data, probs)
        plays = extract_margin_matches(simulation_data)

        _, fig = high_positive_net_strategy(plays,
                                       net_model,
                                       info_data='target set',
                                       save_folder=source_folder)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        response = send_file(img, mimetype='image/png')

    except Exception as e:
        tb = e.__traceback__
        msg = f'{str(e.with_traceback(tb))}'
        logger.error(msg)
        response = make_response(msg, 500)

    return response



