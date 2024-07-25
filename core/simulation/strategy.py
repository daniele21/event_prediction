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


def extract_positive_margin_matches(simulation_data, thr=0):
    margins = ['margin_1',
                 'margin_1X',
                 'margin_2',
                 'margin_X2',
                 'margin_X',
                 'margin_12']
    probs = ["1", '1X', "2", "X2", "X", "12"]
    bets = ['bet_1',
                'bet_X',
                'bet_2',
                'bet_1X',
                'bet_X2',
                'bet_12',]

    plays_dict = {'match_n': [],
                  'margin': [],
                  'choice': [],
                  'bet': [],
                  'kelly_fraction': []}

    for margin, prob, bet in zip(margins, probs, bets):
        matches = simulation_data[simulation_data[margin] > thr].sort_values(by=margin, ascending=False)
        prob_bet_list = matches[['match_n', margin, prob, bet]].to_dict('records')
        for x in prob_bet_list:
            n, p, b, m = x['match_n'], x[prob], x[bet], x[margin]
            kelly = kelly_criterion(b, p)
            plays_dict['match_n'].append(n)
            plays_dict['margin'].append(m)
            plays_dict['kelly_fraction'].append(kelly)
            plays_dict['choice'].append(prob)
            plays_dict['bet'].append(b)

    plays = pd.DataFrame(plays_dict)
    plays = plays.merge(simulation_data[['match_n', 'match_day', 'HomeTeam', 'AwayTeam', 'result_1X2',
                                         ]], on='match_n')

    plays['win'] = plays.apply(lambda x: x['result_1X2'] in x['choice'], axis=1)
    plays = plays[plays['kelly_fraction'] > 0]
    plays['gain'] = ((plays['kelly_fraction'] * plays['bet']) - (plays['kelly_fraction'])) * plays['win'].astype(int)

    return plays



