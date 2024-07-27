from itertools import product

import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
from tqdm import tqdm


def grid_search(estimator, datasets, param_grid, scoring='recall'):
    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    final_scores = pd.DataFrame()

    for params in tqdm(param_combinations, total=len(param_combinations), desc=' > Tuning Hyperparameters: '):
        params_scores = pd.DataFrame()
        model = estimator(**params)

        for i, dataset in datasets.items():
            x_train, x_test = dataset['train']['x'], dataset['test']['x']
            y_train, y_test = dataset['train']['y'], dataset['test']['y']

            # Create and train the model
            model.fit(x_train, y_train.squeeze())

            # Evaluate the model
            predictions = model.predict(x_test)
            recall = recall_score(y_test, predictions, average='weighted')
            precision = precision_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')

            scores = {'estimator': estimator.__name__,
                      **params,
                      'recall': recall,
                      'precision': precision,
                      'f1': f1}
            scores_df = pd.DataFrame(scores, index=[i])
            params_scores = pd.concat((params_scores, scores_df))

        for x in ['recall', 'precision', 'f1']:
            params_scores[f'avg_{x}'] = params_scores[x].mean()

        final_scores = pd.concat((final_scores, params_scores))

    final_scores = final_scores.reset_index().rename(columns={'index': 'target_day'})

    return final_scores
