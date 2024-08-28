from sklearn.model_selection import TimeSeriesSplit

from config.constants import SEASON, MATCH_DAY


def split_data(x, y,
               target_match_day=18,
               test_match_day=2,
               last_n_seasons=5,
               drop_last_seasons=None,
               drop_first_n=0,
               drop_last_n=0):
    drop_last_seasons = None if drop_last_seasons == 0 else -drop_last_seasons

    # Season filter
    seasons = x[SEASON].unique()[-last_n_seasons:drop_last_seasons].tolist()
    print(f'Taking seasons: {seasons}')
    x_train = x[x[SEASON].isin(seasons)]

    # Drop matches
    match_days = x_train[MATCH_DAY].max()
    x_train = x_train[(x_train[MATCH_DAY] > drop_first_n) & \
                      (x_train[MATCH_DAY] <= match_days - drop_last_n)]

    y_train = y.to_frame().loc[x_train.index, :]

    # Split
    last_season = seasons[-1]

    # Target
    x_target = x_train[(x_train[SEASON] == last_season) & \
                       (x_train[MATCH_DAY] == target_match_day)]
    y_target = y_train.loc[x_target.index, :]

    # Test
    x_test = x_train[(x_train[SEASON] == last_season) & \
                      (x_train[MATCH_DAY] >= target_match_day - test_match_day) & \
                      (x_train[MATCH_DAY] < target_match_day)]

    y_test = y_train.loc[x_test.index, :]

    # Train
    x_train = x_train[(x_train[SEASON] < last_season) | \
                     ((x_train[SEASON] == last_season) & \
                      (x_train[MATCH_DAY] < target_match_day - test_match_day))]

    y_train = y_train.loc[x_train.index, :]

    dataset = {'train': {'x': x_train,
                         'y': y_train},
               'test': {'x': x_test,
                        'y': y_test},
               'target': {'x': x_target,
                          'y': y_target}}

    return dataset

def time_series_split(x, y, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y = y.astype('category').cat.codes

    splits = []
    for train_index, test_index in tscv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((x_train, x_test, y_train, y_test))

    return splits
