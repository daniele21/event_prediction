import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score


def training_process(datasets, estimator, params):
    model_scores = pd.DataFrame()
    model_probabilities = pd.DataFrame()
    model = estimator(**params)

    for i, dataset in datasets.items():
        x_train, x_test, x_target = dataset['train']['x'], dataset['test']['x'], dataset['target']['x']
        y_train, y_test, y_target = dataset['train']['y'], dataset['test']['y'], dataset['target']['y']

        x_train = pd.concat((x_train, x_test), axis=0)
        y_train = pd.concat((y_train, y_test), axis=0)

        # Create and train the model
        model.fit(x_train, y_train.squeeze())

        # Evaluate the model
        predictions = model.predict(x_target)
        probabilities = model.predict_proba(x_target)
        recall = recall_score(y_target, predictions, average='weighted')
        precision = precision_score(y_target, predictions, average='weighted')
        f1 = f1_score(y_target, predictions, average='weighted')

        scores = {'estimator': estimator.__name__,
                  **params,
                  'recall': recall,
                  'precision': precision,
                  'f1': f1
                  }
        scores_df = pd.DataFrame(scores, index=[i])
        model_scores = pd.concat((model_scores, scores_df))

        prob_df = pd.concat((x_target, pd.DataFrame(probabilities, index=x_target.index)), axis=1)
        model_probabilities = pd.concat((model_probabilities, prob_df), axis=0)

    model_scores = model_scores.reset_index()\
                                .rename(columns={'index': 'target_day'})
    model_probabilities = model_probabilities.rename(columns={0: 'X', 1:'1', 2:'2'})

    return model, model_scores, model_probabilities
