import numpy as np
import pandas as pd


def preprocessing_season(season_df, n_season, league_name):
    data = season_df.copy(deep=True)

    data.insert(0, 'season', n_season)
    data.insert(0, 'league', league_name)
    data.insert(2, 'match_n', np.arange(1, len(data) + 1, 1))
    # data = _addRound(data)

    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    return data
