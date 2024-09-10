import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def _filter_cols(df):
    cols = ['match_n', 'HomeTeam', 'AwayTeam',
            'result_1X2', 'choice', 'bet',
            'prob', 'bet_prob',
            'kelly', 'ev', 'prob_margin',
            'net', 'win']
    return df[cols]


def plot_scatter_features(data, feature_1, feature_2, hue, size=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature_1, y=feature_2, hue=hue, size=size, palette='coolwarm', alpha=0.7)
    plt.title(f'{feature_1} vs {feature_2}')
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.legend(title=hue)
    plt.show()

# def plot_profit_loss(data, show=True):
#     data['cumulative_net'] = data.sort_index()['net'].cumsum()
#
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data['cumulative_net'])
    plt.title('Profit/Loss Match Days')
    plt.xlabel('Match Day')
    plt.ylabel('Profit/Loss')
#
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#     return fig

def plot_profit_loss(data, show=True):
    net_ratio = (data.groupby('match_day')['net'].sum() / data.groupby('match_day')['kelly'].sum()).\
        to_frame('net_ratio')
    net_ratio['cumulative_net_ratio'] = net_ratio['net_ratio'].expanding().mean()

    fig = plt.figure(figsize=(10, 6))
    plt.bar(net_ratio.index, net_ratio['net_ratio'].to_list(), alpha=0.5, color='b', label='net ratio')
    plt.plot(net_ratio.index, net_ratio['cumulative_net_ratio'], label='cumulative net ratio')
    plt.title('Net Ratio per Match Days')
    plt.xlabel('Match Day')
    plt.ylabel('Net/Spent')
    plt.legend()
    plt.grid()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def best_ev(results):
    df = _filter_cols(results)
    df = df.sort_values(by='ev', ascending=False) \
        .drop_duplicates()
    df


def predict_net_revenue(test_df, target_df):
    features = ['ev', 'kelly', 'prob_margin']

    test_df = test_df[test_df['kelly'] > 0]
    target_df = target_df[target_df['kelly'] > 0]

    df = test_df[features]
    df['target'] = test_df['net']

    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=2024)

    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 8, 13, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        # 'max_features': ['sqrt', 'log2', None]
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV with the Random Forest and parameter grid
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=3,
                               scoring='neg_mean_absolute_error',
                               n_jobs=-1,
                               verbose=1)

    # Perform grid search on the filtered training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score from the grid search
    best_params = grid_search.best_params_

    params = best_params
    # params = {'n_estimators': 300,
    #           'max_depth': 5,
    #           'random_state': 2024}
    model = RandomForestRegressor(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(mae)

    X_test = X_test.merge(test_df['net'], how='left', left_index=True, right_index=True)
    X_test.loc[:, 'predicted_net'] = y_pred
    X_test['bet_decision'] = X_test['predicted_net'].apply(lambda x: 1 if x > 0 else 0)
    X_test = X_test.sort_index()

    plot_profit_loss(X_test[X_test['bet_decision'] == 1])

    y_target = target_df['net']
    y_pred_target = model.predict(target_df[features])

    mae_target = mean_absolute_error(y_target, y_pred_target)
    print(mae_target)

    target_df.loc[:, 'predicted_net'] = y_pred_target
    target_df['bet_decision'] = target_df['predicted_net'].apply(lambda x: 1 if x > 0 else 0)
    target_df = target_df.sort_index()

    plot_profit_loss(target_df[target_df['bet_decision'] == 1])


def learn_decision_boundary(test_df, target_df):
    features = ['ev', 'kelly', 'prob_margin']

    test_df = test_df[test_df['kelly'] > 0]
    target_df = target_df[target_df['kelly'] > 0]

    df = test_df[features]
    df['target'] = test_df['net'].apply(lambda x: 1 if x > 0 else 0)

    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=2024)

    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 8, 13, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        #'max_features': ['sqrt', 'log2', None]
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV with the Random Forest and parameter grid
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=3,
                               scoring='precision',
                               n_jobs=-1,
                               verbose=1)

    # Perform grid search on the filtered training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score from the grid search
    best_params = grid_search.best_params_

    params = best_params
    # params = {'n_estimators': 300,
    #           'max_depth': 5,
    #           'random_state': 2024}
    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    classification_report_test = classification_report(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred_prob)

    print(classification_report_test)
    print(roc_auc_test)

    y_test_pred_df = pd.DataFrame(y_pred)
    test_df.loc[:, 'bet_decision'] = y_test_pred_df
    # test_df.loc[test_df['kelly'] < 0, 'bet_decision'] = 0
    plot_profit_loss(test_df[test_df['bet_decision'] == 1])

    # Backtesting
    backtest_df = target_df[features].copy(deep=True)
    backtest_df['target'] = target_df['net'].apply(lambda x: 1 if x > 0 else 0)

    X_target = backtest_df[features]
    y_target = backtest_df['target']

    y_target_pred = model.predict(X_target)
    y_target_pred_prob = model.predict_proba(X_target)[:, 1]

    classification_report_target = classification_report(y_target, y_target_pred)
    roc_auc_target = roc_auc_score(y_target, y_target_pred_prob)

    print(classification_report_target)
    print(roc_auc_target)

    y_target_pred_df = pd.DataFrame(y_target_pred)
    target_df.loc[:, 'bet_decision'] = y_target_pred_df
    # target_df.loc[target_df['kelly']<0, 'bet_decision'] = 0

    plot_profit_loss(target_df[target_df['bet_decision'] == 1])

    return


if __name__ == '__main__':
    training_results_path = 'outputs/training_test/serie_a/target_2324_test_model_20240825_222043.xlsx'
    test_results = pd.read_excel(training_results_path, sheet_name='test').drop_duplicates()
    target_results = pd.read_excel(training_results_path, sheet_name='target')

    predict_net_revenue(test_results, target_results)

    learn_decision_boundary(test_results, target_results)

    best_ev(target_results)
    best_ev(test_results)
