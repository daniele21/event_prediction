

def enrich_data_for_simulation(data, probabilities):
    result = probabilities.merge(data[['match_n', 'Date',
                                       'HomeTeam', 'AwayTeam', 'result_1X2',
                                       'bet_1', 'bet_X', 'bet_2']],
                                    left_index=True,
                                    right_index=True,
                                    )
    result['1X'] = result['1'] + result['X']
    result['X2'] = result['2'] + result['X']
    result['12'] = result['1'] + result['2']
    result['bet_1X'] = 1/((1/result['bet_1']) + (1/result['bet_X']))
    result['bet_X2'] = 1/((1/result['bet_X']) + (1/result['bet_2']))
    result['bet_12'] = 1/((1/result['bet_1']) + (1/result['bet_2']))
    result['my_bet_1'] = 1 / result['1']
    result['my_bet_X'] = 1 / result['X']
    result['my_bet_2'] = 1 / result['2']
    result['my_bet_1X'] = 1/result['1X']
    result['my_bet_X2'] = 1 / result['X2']
    result['my_bet_12'] = 1 / result['12']

    result['bet_margin_1'] = result['bet_1'] - result['my_bet_1']
    result['bet_margin_1X'] = result['bet_1X'] - result['my_bet_1X']
    result['bet_margin_2'] = result['bet_2'] - result['my_bet_2']
    result['bet_margin_X2'] = result['bet_X2'] - result['my_bet_X2']
    result['bet_margin_X'] = result['bet_X'] - result['my_bet_X']
    result['bet_margin_12'] = result['bet_12'] - result['my_bet_12']

    result['prob_margin_1'] = result['1'] - (1/result['bet_1'])
    result['prob_margin_1X'] = result['1X'] - (1/result['bet_1X'])
    result['prob_margin_2'] = result['2'] - (1/result['bet_2'])
    result['prob_margin_X2'] = result['X2'] - (1/result['bet_X2'])
    result['prob_margin_X'] = result['X'] - (1/result['bet_X'])
    result['prob_margin_12'] = result['12'] - (1/result['bet_12'])

    result = result[['season', 'match_day', 'match_n',
                     'Date', 'HomeTeam', 'AwayTeam',
                     'result_1X2',
                     'bet_margin_1',
                     'prob_margin_1',
                     'bet_margin_1X',
                     'prob_margin_1X',
                     'bet_margin_2',
                     'prob_margin_2',
                     'bet_margin_X2',
                     'prob_margin_X2',
                     'bet_margin_X',
                     'prob_margin_X',
                     'bet_margin_12',
                     'prob_margin_12',
                     "1", 'X', "2", "1X", "X2", "12",
                     'bet_1', 'my_bet_1',
                     'bet_X', 'my_bet_X',
                     'bet_2', 'my_bet_2',
                     'bet_1X', 'my_bet_1X',
                     'bet_X2', 'my_bet_X2',
                     'bet_12', 'my_bet_12',
    ]]

    return result
