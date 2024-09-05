from config.constants import MATCH_RESULT_V1, MATCH_RESULT_V2, MATCH_RESULT_V1_1


def dataset_preprocessing(dataset_version):
    if dataset_version == MATCH_RESULT_V1:
        return match_result_v1
    elif dataset_version == MATCH_RESULT_V1_1:
        return match_result_v1_1
    elif dataset_version == MATCH_RESULT_V2:
        return match_result_v2
    else:
        raise AttributeError(f'No dataset preprocessing version found for >> {dataset_version} << ')

def encode_match_result(result_1x2):
    if str(result_1x2) == "1":
        return 1
    elif str(result_1x2) == "2":
        return 2
    elif str(result_1x2) == "X":
        return 0
    elif str(result_1x2) == 'UNKNOWN':
        return -1
    else:
        raise AttributeError(f'No match result value found for >> {result_1x2} << ')
def match_result_v1(data):
    target = 'result_1X2'
    drop_cols = ['home_goals',
                  'away_goals',
                  'league',
                  'AwayTeam', 'HomeTeam',
                  'Date', 'match_n', 'bet_1',
                  'bet_X', 'bet_2']

    data = data.drop(drop_cols, axis=1)

    x = data.drop(target, axis=1).fillna(0)
    y = data[target].apply(encode_match_result)

    return x, y

def match_result_v1_1(data):
    target = 'result_1X2'
    drop_cols = ['home_goals',
                  'away_goals',
                 'season',
                 'match_day',
                  'league',
                  'AwayTeam', 'HomeTeam',
                  'Date', 'match_n', 'bet_1',
                  'bet_X', 'bet_2']

    data = data.drop(drop_cols, axis=1)

    x = data.drop(target, axis=1).fillna(0)
    y = data[target].apply(encode_match_result)

    return x, y

def match_result_v2(data):
    target = 'result_1X2'
    drop_cols = ['home_goals',
                  'away_goals',
                  'league',
                  'AwayTeam', 'HomeTeam',
                  'Date', 'match_n']

    data = data.drop(drop_cols, axis=1)

    x = data.drop(target, axis=1).fillna(0)
    y = data[target].apply(encode_match_result)

    return x, y

