import pandas as pd

from core.strategies.kelly import kelly_criterion


def extract_margin_matches(simulation_data, margin_thr=-100, kelly_thr=-100):
    events = ["1", '1X', "2", "X2", "X", "12"]
    bet_margins = [f'bet_margin_{event}' for event in events]
    prob_margins = [f'prob_margin_{event}' for event in events]
    bets = [f'bet_{event}' for event in events]

    plays_dict = {'match_day': [],
                  'match_n': [],
                  'bet_margin': [],
                  'prob_margin': [],
                  'choice': [],
                  'bet': [],
                  'my_prob_kelly': [],
                  'bet_prob_kelly': [],
                  'prob': []}

    for bet_margin, prob_margin, event, bet in zip(bet_margins, prob_margins, events, bets):

        event_bet_list = simulation_data[['match_day', 'match_n',
                                          bet_margin, prob_margin,
                                          event, bet]].to_dict('records')

        for x in event_bet_list:
            match_day, match_n = x['match_day'], x['match_n']
            prob, bet_item, bet_margin_item, prob_margin_item = x[event], x[bet], x[bet_margin], x[prob_margin]

            my_kelly = kelly_criterion(bet_item, prob)
            bet_kelly = kelly_criterion(bet_item, 1/bet_item)
            plays_dict['match_day'].append(match_day)
            plays_dict['match_n'].append(match_n)
            plays_dict['bet_margin'].append(bet_margin_item)
            plays_dict['prob_margin'].append(prob_margin_item)
            plays_dict['my_prob_kelly'].append(my_kelly)
            plays_dict['bet_prob_kelly'].append(bet_kelly)
            plays_dict['choice'].append(event)
            plays_dict['bet'].append(bet_item)
            plays_dict['prob'].append(prob)

    plays = pd.DataFrame(plays_dict)
    plays = plays.merge(simulation_data[['match_day', 'match_n', 'HomeTeam', 'AwayTeam', 'result_1X2',
                                         ]], on=['match_day', 'match_n'])

    plays['win'] = plays.apply(lambda x: x['result_1X2'] in x['choice'], axis=1)

    plays['my_prob_gain'] = (plays['my_prob_kelly'] * plays['bet']) * plays['win'].astype(int)
    plays['my_prob_spent'] = plays['my_prob_kelly']
    plays['my_prob_net'] = plays['my_prob_gain'] - plays['my_prob_spent']
    plays['my_prob_net_rate'] = plays['my_prob_net'] / plays['my_prob_spent']

    plays['bet_prob_gain'] = (plays['bet_prob_kelly'] * plays['bet']) * plays['win'].astype(int)
    plays['bet_prob_spent'] = plays['bet_prob_kelly']
    plays['bet_prob_net'] = plays['bet_prob_gain'] - plays['bet_prob_spent']
    plays['bet_prob_net_rate'] = plays['bet_prob_net'] / plays['bet_prob_spent']

    return plays
