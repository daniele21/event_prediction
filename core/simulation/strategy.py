import pandas as pd


def kelly_criterion(bet, prob):
    """Calculate the optimal Kelly betting fraction.

    Args:
        bet (float): The net odds received on the wager ("b to 1").
        prob (float): The probability of winning.

    Returns:
        float: The fraction of the bankroll to bet, according to the Kelly Criterion.
    """
    q = 1 - prob
    return (bet * prob - q) / bet


def extract_positive_margin_matches(simulation_data, margin_thr=0):
    events = ["1", '1X', "2", "X2", "X", "12"]
    margins = [f'margin_{event}' for event in events]
    bets = [f'bet_{event}' for event in events]

    plays_dict = {'match_day': [],
                  'match_n': [],
                  'margin': [],
                  'choice': [],
                  'bet': [],
                  'kelly_fraction': []}

    for margin, event, bet in zip(margins, events, bets):
        matches = simulation_data[simulation_data[margin] > margin_thr]\
                    .sort_values(by=margin, ascending=False)

        event_bet_list = matches[['match_day', 'match_n',
                                 margin, event, bet]].to_dict('records')

        for x in event_bet_list:
            match_day, match_n = x['match_day'], x['match_n']
            prob, bet_item, margin_item = x[event], x[bet], x[margin]

            kelly = kelly_criterion(bet_item, prob)
            plays_dict['match_day'].append(match_day)
            plays_dict['match_n'].append(match_n)
            plays_dict['margin'].append(margin_item)
            plays_dict['kelly_fraction'].append(kelly)
            plays_dict['choice'].append(event)
            plays_dict['bet'].append(bet_item)

    plays = pd.DataFrame(plays_dict)
    plays = plays.merge(simulation_data[['match_day', 'match_n', 'HomeTeam', 'AwayTeam', 'result_1X2',
                                         ]], on=['match_day', 'match_n'])

    plays['win'] = plays.apply(lambda x: x['result_1X2'] in x['choice'], axis=1)
    plays = plays[plays['kelly_fraction'] > 0]
    plays['gain'] = ((plays['kelly_fraction'] * plays['bet']) - (plays['kelly_fraction'])) * plays['win'].astype(int)

    return plays
