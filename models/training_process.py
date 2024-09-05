import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss
from tqdm import tqdm

from scripts.training.validation import check_calibration


def train_and_test(x_train, x_test, y_train, y_test,
                   estimator, params):
    if estimator.__name__ == 'LGBMClassifier':
        lgbm_params = {k: v for k, v in params.items() if k != 'num_iterations'}
        model = estimator(**lgbm_params)

        model.fit(x_train, y_train.squeeze(),
                  eval_set=[(x_train, y_train), (x_test, y_test)])
    else:
        model = estimator(**params)
        model.fit(x_train, y_train.squeeze())

    return model


def train_and_target(x_train, x_test, y_train, y_test,
                     estimator, params):
    x_train = pd.concat((x_train, x_test), axis=0)
    y_train = pd.concat((y_train, y_test), axis=0)

    if estimator.__name__ == 'LGBMClassifier':
        lgbm_params = {k: v for k, v in params.items() if k != 'early_stopping_round'}
        model = estimator(**lgbm_params)

        model.fit(x_train, y_train.squeeze())
    else:
        model = estimator(**params)
        model.fit(x_train, y_train.squeeze())

    return model


def new_training_process(datasets, estimator, params, scoring, inference=False):
    model_scores = pd.DataFrame()
    model_probabilities = pd.DataFrame()
    test_model_probabilities = pd.DataFrame()
    model_on_test_dict = {}
    model_on_target_dict = {}

    for i_day, dataset in tqdm(datasets.items(),
                               desc=' > Training Steps: '):
        x_train, x_test, x_target = dataset['train']['x'], dataset['test']['x'], dataset['target']['x']
        y_train, y_test, y_target = dataset['train']['y'], dataset['test']['y'], dataset['target']['y']

        x_train = x_train.drop(['match_day', 'season'], axis=1)
        x_test = x_test.drop(['match_day', 'season'], axis=1)
        x_target = x_target.drop(['match_day', 'season'], axis=1)

        model_on_test = train_and_test(x_train, x_test, y_train, y_test,
                                       estimator, params[scoring][i_day])
        model_on_target = train_and_target(x_train, x_test, y_train, y_test,
                                           estimator, params[scoring][i_day])

        model_on_test_dict[i_day] = model_on_test
        model_on_target_dict[i_day] = model_on_target

        # Evaluate the model
        # test_predictions = model_on_test.predict(x_test)
        test_probabilities = model_on_test.predict_proba(x_test)
        test_prob = pd.DataFrame(test_probabilities, index=x_test.index)
        test_prob['target_day'] = i_day
        # test_recall = recall_score(y_test, test_predictions, average='weighted')
        # test_precision = precision_score(y_test, test_predictions, average='weighted')
        # test_f1 = f1_score(y_test, test_predictions, average='weighted')
        # test_ll = -log_loss(y_test, test_probabilities, labels=[0, 1, 2])

        target_predictions = model_on_test.predict(x_target)
        target_probabilities = model_on_test.predict_proba(x_target)
        target_recall = recall_score(y_target, target_predictions, average='weighted')
        target_precision = precision_score(y_target, target_predictions, average='weighted')
        target_f1 = f1_score(y_target, target_predictions, average='weighted')
        target_ll = -log_loss(y_target, target_probabilities, labels=[0, 1, 2])

        scores = {'estimator': estimator.__name__,
                  **params[scoring][i_day],
                  'recall': target_recall,
                  'precision': target_precision,
                  'f1': target_f1,
                  'll': target_ll
                  }
        scores_df = pd.DataFrame(scores, index=[i_day])
        model_scores = pd.concat((model_scores, scores_df))

        prob_df = pd.concat((x_target, pd.DataFrame(target_probabilities, index=x_target.index)), axis=1)
        test_prob_df = pd.concat((x_test, test_prob), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)
        test_model_probabilities = pd.concat((test_model_probabilities, test_prob_df), axis=0)

    model_scores = model_scores.reset_index() \
        .rename(columns={'index': 'target_day'})
    model_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})
    test_model_probabilities = test_model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    return model_on_target_dict, model_scores, model_probabilities, test_model_probabilities


def training_process(datasets, estimator, params, scoring, inference=False):
    model_scores = pd.DataFrame()
    model_probabilities = pd.DataFrame()
    test_model_probabilities = pd.DataFrame()
    model_dict = {}
    for i_day, dataset in tqdm(datasets.items(),
                               desc=' > Training Steps: '):
        x_train, x_test, x_target = dataset['train']['x'], dataset['test']['x'], dataset['target']['x']
        y_train, y_test, y_target = dataset['train']['y'], dataset['test']['y'], dataset['target']['y']

        model = estimator(**params[scoring][i_day])

        if inference:
            x_train = pd.concat((x_train, x_test), axis=0)
            y_train = pd.concat((y_train, y_test), axis=0)

        # Create and train the model
        if estimator.__name__ == 'LGBMClassifier':
            if inference:
                lgbm_params = {k: v for k, v in params[scoring][i_day].items() if k != 'early_stopping_round'}
                model = estimator(**lgbm_params)
            else:
                lgbm_params = {k: v for k, v in params[scoring][i_day].items() if k != 'num_iterations'}
                model = estimator(**lgbm_params)

            model.fit(x_train, y_train.squeeze(),
                      eval_set=[(x_train, y_train), (x_test, y_test)])
        else:
            model.fit(x_train, y_train.squeeze())

        model_dict[i_day] = model

        # Evaluate the model
        x_valid_set = x_target if inference else x_test
        y_valid_set = y_target if inference else y_test
        predictions = model.predict(x_valid_set)
        probabilities = model.predict_proba(x_valid_set)
        recall = recall_score(y_valid_set, predictions, average='weighted')
        precision = precision_score(y_valid_set, predictions, average='weighted')
        f1 = f1_score(y_valid_set, predictions, average='weighted')
        ll = -log_loss(y_valid_set, probabilities, labels=[0, 1, 2])

        test_probabilities = model.predict_proba(x_test)

        scores = {'estimator': estimator.__name__,
                  **params[scoring][i_day],
                  'recall': recall,
                  'precision': precision,
                  'f1': f1,
                  'll': ll
                  }
        scores_df = pd.DataFrame(scores, index=[i_day])
        model_scores = pd.concat((model_scores, scores_df))

        prob_df = pd.concat((x_valid_set, pd.DataFrame(probabilities, index=x_valid_set.index)), axis=1)
        test_prob_df = pd.concat((x_test, pd.DataFrame(test_probabilities, index=x_test.index)), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)
        test_model_probabilities = pd.concat((test_model_probabilities, test_prob_df), axis=0)

    model_scores = model_scores.reset_index() \
        .rename(columns={'index': 'target_day'})
    model_probabilities = model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})
    test_model_probabilities = test_model_probabilities.rename(columns={0: 'X', 1: '1', 2: '2'})

    return model_dict, model_scores, model_probabilities, test_model_probabilities
