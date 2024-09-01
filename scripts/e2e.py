import pandas as pd
from sklearn.metrics import mean_absolute_error, recall_score, precision_score, f1_score, log_loss, \
    classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from core.dataset.dataset_generation import generate_datasets
from core.grid_search import grid_search_regression
from core.ingestion.update_league_data import get_most_recent_data
from core.logger import logger
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_margin_matches
from core.utils import load_json, ensure_folder, save_json, get_timestamp, save_pickle
from models.training_framework import hyperparameter_tuning, get_estimator
from models.training_process import training_process
from scripts.strategy import plot_profit_loss
from scripts.training.validation import check_calibration


def ht_section(data, dataset_params, training_params, save_folder=None):
    datasets = generate_datasets(data, dataset_params)
    estimator, best_params_df, perf_scores = hyperparameter_tuning(datasets, training_params)

    metric = training_params['scoring']
    best_params = best_params_df.T[metric].to_dict()

    if save_folder:
        print(f'Saving Tuning results at: {save_folder}')
        dataset_path = f'{save_folder}/dataset_config.json'
        tuning_path = f'{save_folder}/tuning_config.json'
        save_json(dataset_params, dataset_path)
        save_json(training_params, tuning_path)

        best_params_path = f'{save_folder}/tuning_best_params.json'
        tuning_scores_path = f'{save_folder}/tuning_scores.csv'
        save_json(best_params_df.reset_index().to_dict('records'), best_params_path)
        perf_scores.to_csv(tuning_scores_path)

    return best_params


def training_and_test_section(data, dataset_params, training_params, params, save_folder=None):
    datasets = generate_datasets(data, dataset_params)
    estimator = get_estimator(training_params['estimator'])

    model, model_scores, probabilities = training_process(datasets, estimator, params, inference=False)
    sub_data = data[['league', 'season', 'match_day', 'match_n', 'Date', 'HomeTeam', 'AwayTeam',
                     'result_1X2', 'bet_1', 'bet_X', 'bet_2']]
    prediction = sub_data.merge(probabilities[['1', 'X', '2']], how='right', left_index=True, right_index=True)

    # Simulation
    simulation_data = enrich_data_for_simulation(data, probabilities)
    bet_plays = extract_margin_matches(simulation_data)

    fig = check_calibration(bet_plays, prob_col='prob', title='Test Calibration', show=False)

    if save_folder:
        print(f'Saving Training results at: {save_folder}')

        model_path = f'{save_folder}/class_model.pkl'
        logger.info(f' > Saving Classification Model at {model_path}')
        save_pickle(model, model_path)

        model_scores_path = f'{save_folder}/model_scores.csv'
        logger.info(f' > Saving Classification Model scores at {model_scores_path}')
        model_scores.to_csv(model_scores_path)

        prediction_path = f'{save_folder}/model_prediction.csv'
        logger.info(f' > Saving Classification Prediction at {prediction_path}')
        prediction.to_csv(prediction_path)

        bet_plays_path = f'{save_folder}/bet_plays.csv'
        logger.info(f' > Saving bet plays at {bet_plays_path}')
        bet_plays.to_csv(bet_plays_path)

        calibration_path = f'{save_folder}/test_calibration.png'
        logger.info(f' > Saving model calibration at {calibration_path}')
        fig.savefig(calibration_path)

    return model, bet_plays, fig


def net_prediction_model(bet_plays, strategy_config, save_folder=None):
    df = bet_plays.copy(deep=True)
    df = df[df['kelly'] > 0].drop_duplicates()

    features = ['ev', 'prob_margin', 'kelly']
    target = ['net']

    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=strategy_config['test_size'],
                                                        random_state=2024)
    print(f'Strategy train test split: {x_train.shape} - {x_test.shape} | {y_train.shape} - {y_test.shape}')
    dataset = {'train': {'x': x_train,
                         'y': y_train},
               'test': {'x': x_test,
                        'y': y_test}
               }

    estimator, best_params = grid_search_regression(dataset, strategy_config)

    net_model = estimator(**best_params)

    if estimator.__name__ == 'LGBMRegressor':
        net_model.fit(x_train, y_train.squeeze(),
                      eval_set=[(x_train, y_train), (x_test, y_test)])
    else:
        net_model.fit(x_train, y_train.squeeze())

    y_train_pred = net_model.predict(x_train)
    y_test_pred = net_model.predict(x_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f'Net Regression MAE on train set: {train_mae}')
    print(f'Net Regression MAE on test set: {test_mae}')

    x_test = df[['match_day']].merge(x_test, how='right', left_index=True, right_index=True)
    x_test = x_test.merge(df[['net', 'win']], how='left', left_index=True, right_index=True)
    x_test.loc[:, 'predicted_net'] = y_test_pred

    if save_folder:
        # pl_path = f'{save_folder}/pl_strategy_on_test_set.png'
        # logger.info(f' > Saving Profit/Loss on test set at {pl_path}')
        # fig.savefig(pl_path)

        # test_bet_path = f'{save_folder}/bet_decision_test.csv'
        # logger.info(f' > Saving bet decision test set at {test_bet_path}')
        # x_test.to_csv(test_bet_path)

        # class_report_path = f'{save_folder}/bet_decision_class_report_test.csv'
        # logger.info(f' > Saving bet decision classification report test set at {class_report_path}')
        # pd.DataFrame(class_report).T.to_csv(class_report_path)

        net_model_path = f'{save_folder}/net_prediction_model.pkl'
        logger.info(f' > Saving Net Model at {net_model_path}')
        save_pickle(net_model, net_model_path)

    return net_model, x_test


def backtesting(data, dataset_params, class_model, net_model, save_folder=None):
    model_probabilities = pd.DataFrame()
    model_scores = pd.DataFrame()

    datasets = generate_datasets(data, dataset_params)

    for i, dataset in datasets.items():
        x_target = dataset['target']['x']
        y_target = dataset['target']['y']

        predictions = class_model.predict(x_target)
        probabilities = class_model.predict_proba(x_target)
        recall = recall_score(y_target, predictions, average='weighted')
        precision = precision_score(y_target, predictions, average='weighted')
        f1 = f1_score(y_target, predictions, average='weighted')
        ll = log_loss(y_target, probabilities, labels=[0, 1, 2])

        scores = {'recall': recall,
                  'precision': precision,
                  'f1': f1,
                  'log_loss': ll,
                  }
        scores_df = pd.DataFrame(scores, index=[i])
        model_scores = pd.concat((model_scores, scores_df))

        prob_df = pd.concat((x_target, pd.DataFrame(probabilities, index=x_target.index)), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)

    target_scores = model_scores.reset_index() \
        .rename(columns={'index': 'target_day'})
    target_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    sub_data = data[['league', 'season', 'match_day', 'match_n', 'Date', 'HomeTeam', 'AwayTeam',
                     'result_1X2', 'bet_1', 'bet_X', 'bet_2']]
    prediction = sub_data.merge(target_probabilities[['1', 'X', '2']], how='right', left_index=True, right_index=True)

    # Simulation
    simulation_data = enrich_data_for_simulation(data, target_probabilities)
    target_plays = extract_margin_matches(simulation_data)

    df = target_plays.copy(deep=True)
    df = df[df['kelly'] > 0]

    features = ['ev', 'prob_margin', 'kelly']
    target = ['net']

    x_target, y_target = df[features], df[target]

    y_target_pred = net_model.predict(x_target)

    mae = mean_absolute_error(y_target, y_target_pred)
    print(f'Net Regression MAE on target set: {mae}')

    x_target = x_target.merge(df[['net', 'win']], how='left', left_index=True, right_index=True)
    x_target.loc[:, 'predicted_net'] = y_target_pred
    x_target['bet_decision'] = x_target['predicted_net'].apply(lambda x: 1 if x > 0 else 0)
    x_target = x_target.sort_index()

    class_report = classification_report(x_target['win'].astype(int), x_target['bet_decision'],
                                         output_dict=True)
    print(f'Bet Decision Classification Report | Target')
    print(classification_report(x_target['win'].astype(int), x_target['bet_decision']))

    fig = plot_profit_loss(x_target[x_target['bet_decision'] == 1], show=False)

    if save_folder:
        pl_path = f'{save_folder}/pl_strategy_on_target_set.png'
        fig.savefig(pl_path)

        target_bet_path = f'{save_folder}/bet_decision_target.csv'
        x_target.to_csv(target_bet_path)

        class_report_path = f'{save_folder}/bet_decision_class_report_target.csv'
        pd.DataFrame(class_report).T.to_csv(class_report_path)

        prediction_path = f'{save_folder}/target_prediction.csv'
        prediction.to_csv(prediction_path)


if __name__ == '__main__':
    exp_prefix = 'last_7_seasons-2'

    dataset_config = load_json('config/dataset.json')
    training_config = load_json('config/training_lgbm.json')
    strategy_config = load_json('config/strategy_rf.json')

    source_data = get_most_recent_data(dataset_config)

    e2e_folder = f'outputs/e2e/{dataset_config["league_name"]}/{exp_prefix}_{get_timestamp()}'
    ensure_folder(e2e_folder)

    # Hyperparameter Tuning
    best_params = ht_section(source_data,
                             dataset_config,
                             training_config,
                             save_folder=e2e_folder)

    # Training & Test
    class_model, bet_plays, _ = training_and_test_section(source_data,
                                                       dataset_config,
                                                       training_config,
                                                       best_params,
                                                       save_folder=e2e_folder)

    # Strategy
    net_model = net_prediction_model(bet_plays,
                                     strategy_config,
                                     save_folder=e2e_folder)

    # Backtesting
    backtesting(source_data,
                dataset_config,
                class_model,
                net_model,
                save_folder=e2e_folder)
