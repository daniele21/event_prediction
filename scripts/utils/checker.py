from json import JSONDecodeError
from os.path import exists

from config.league import LEAGUE_NAMES


def check_league(league_name):
    league_name = str(league_name).lower()

    if(league_name not in LEAGUE_NAMES):
        msg = f' League not found: >>> {league_name} <<<'
        check = False
    else:
        msg = f' League found: {league_name}'
        check = True

    response = {'check':check,
                'msg':msg}

    return response

def check_data_league(league_name, npm, data_dir):

    data_path = f'{data_dir}{league_name}/{league_name}_npm={npm}.csv'

    check = exists(data_path)
    if not check:
        msg = f'Data not Found: {data_path}'
    else:
        msg = f'Data Found at {data_path}'

    response = {'check':check,
                'msg':msg}

    return response

def check_npm(npm):

    try:
        check = True if int(npm) > 0 else False
        msg = 'Valid NPM parameter passed'

    except ValueError:
        check = False
        msg = f'Invalid NPM parameter passed: {msg} is not castable to an integer number'

    response = {'check':check,
                'msg':msg}

    return response

def check_training_args(args):

    check = 'epochs' in list(args.keys()) and \
            'patience' in list(args.keys())

    if not check:
        msg = f'Args not found: {list(args.keys())}'
    else:
        msg = 'Args read'

    return {'check':check,
            'msg':msg}


def check_training_params(params):
    check = ['league', 'data', 'model'] == list(params.keys())

    if not check:
        msg = f'Params not found: {params.keys()}'
    else:
        msg = 'Params read'

    return {'check': check,
            'msg': msg}

