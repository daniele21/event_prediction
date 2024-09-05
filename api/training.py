import io
import json

from flask import current_app as app, render_template_string, send_file
from flask import make_response, jsonify, request, render_template
import json

from api.templates.table_template import HTML_TEMPLATE
from core.ingestion.update_league_data import update_data_league, get_most_recent_data
from core.logger import logger
from core.utils import get_timestamp, ensure_folder, load_json
from scripts.new_e2e import ht_section, training_and_test_section


@app.route('/api/training/hyperparameter_tuning', methods=['POST'])
def hyperparameter_tuning():
    payload = request.json
    exp_name = payload['exp_name']
    dataset_params = payload['dataset']
    training_params = payload['training']
    source_folder = payload.get('source_folder')

    e2e_folder = f'outputs/e2e/{dataset_params["league_name"]}/{exp_name}_{get_timestamp()}'
    source_folder = e2e_folder if source_folder is None else source_folder

    ensure_folder(source_folder)

    source_data = get_most_recent_data(dataset_params)

    # try:
    best_params_dict = ht_section(source_data, dataset_params, training_params, save_folder=source_folder)
    best_params_str = str(best_params_dict).replace("'", '"')
    msg = f"Tuning results: \n\nModel: {training_params['estimator']}\n\n{best_params_str}\nOptimized score: {training_params['scoring']}\n\n"
    msg += f"Saved at\n{source_folder}"
    response = make_response(msg, 200)
    # except Exception as e:
    #     tb = e.__traceback__
    #     msg = f'{str(e.with_traceback(tb))}'
    #     logger.error(msg)
    #     response = make_response(msg, 500)

    return response


@app.route('/api/training', methods=['POST'])
def training():
    payload = request.json
    exp_name = payload.get('exp_name')
    dataset_params = payload.get('dataset')
    training_params = {'estimator': payload['estimator']} if 'estimator' in payload else None
    params = payload.get('params')
    source_folder = payload.get('source_folder')

    if dataset_params is None:
        dataset_path = f'{source_folder}/dataset_config.json'
        dataset_params = load_json(dataset_path)
        logger.info(dataset_params)

    if params is None and training_params is None:
        tuning_path = f'{source_folder}/tuning_config.json'
        params_path = f'{source_folder}/tuning_best_params.json'
        tuning_config = load_json(tuning_path)
        params = load_json(params_path)
        for metric, value in params.items():
            params[metric] = {}
            for k, v in value.items():
                params[metric][int(k)] = v

        # best_params = [x for x in params if x['index'] == tuning_config['scoring']][0]
        # best_params.pop('index')
        training_params = {'estimator': tuning_config['estimator'],
                           'scoring': tuning_config['scoring'],
                           "splits": tuning_config['splits'],
                           "max_train_size": tuning_config['max_train_size'],
                           "test_size": tuning_config['test_size']
                           }
        logger.info(params)

    e2e_folder = f'outputs/e2e/{dataset_params["league_name"]}/{exp_name}_{get_timestamp()}'
    source_folder = e2e_folder if source_folder is None else source_folder

    ensure_folder(source_folder)

    source_data = get_most_recent_data(dataset_params)

    try:
        class_model, bet_plays, test_bet_plays, fig = training_and_test_section(source_data,
                                                                                dataset_params,
                                                                                training_params,
                                                                                params,
                                                                                save_folder=source_folder)

        # importances = pd.DataFrame()
        # for day, model in class_model.items():
        #     a = pd.DataFrame(model.feature_importances_, model.feature_name_).rename(columns={0: 'importance'}).T
        #     a['target_day'] = day
        #     a = a.set_index('target_day')
        #     importances = pd.concat((importances, a))

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
