import pandas as pd

from core.strategies.kelly import kelly_criterion


def extract_margin_matches(simulation_data, margin_thr=-100, kelly_thr=-100):
    events = ["1", '1X', "2", "X2", "X", "12"]
    bet_margins = [f'bet_margin_{event}' for event in events]
    prob_margins = [f'prob_margin_{event}' for event in events]
    bets = [f'bet_{event}' for event in events]

    plays_dict = {'match_day': [],
                  'match_n': [],
                  'choice': [],
                  'bet': [],
                  'bet_prob': [],
                  'prob': [],
                  'prob_margin': [],
                  'kelly': []
                  }

    for bet_margin, prob_margin, event, bet in zip(bet_margins, prob_margins, events, bets):

        event_bet_list = simulation_data[['match_day', 'match_n',
                                          bet_margin, prob_margin,
                                          event, bet]].to_dict('records')

        for x in event_bet_list:
            match_day, match_n = x['match_day'], x['match_n']
            prob, bet_item, bet_margin_item, prob_margin_item = x[event], x[bet], x[bet_margin], x[prob_margin]

            kelly = kelly_criterion(bet_item, prob)
            plays_dict['match_day'].append(match_day)
            plays_dict['match_n'].append(match_n)
            plays_dict['prob_margin'].append(prob_margin_item)
            plays_dict['kelly'].append(kelly)
            plays_dict['choice'].append(event)
            plays_dict['bet'].append(bet_item)
            plays_dict['prob'].append(prob)
            plays_dict['bet_prob'].append(1 / bet_item)

    cols = ['league', 'season', 'match_day', 'match_n', 'HomeTeam', 'AwayTeam', 'result_1X2']
    plays = pd.DataFrame(plays_dict)
    # plays = plays.merge(simulation_data[['match_day', 'match_n', 'HomeTeam', 'AwayTeam', 'result_1X2',
    #                                      ]], on=['match_day', 'match_n'])
    plays = simulation_data[cols].merge(plays, how='right', on=['match_day', 'match_n'])

    plays['ev'] = (plays['bet'] * plays['prob']) - 1

    plays['win'] = plays.apply(lambda x: x['result_1X2'] in x['choice'], axis=1)

    plays['gain'] = (plays['kelly'] * plays['bet']) * plays['win'].astype(int)
    plays['spent'] = plays['kelly']
    plays['net'] = plays['gain'] - plays['spent']
    plays['net_rate'] = plays['net'] / plays['spent']

    return plays
