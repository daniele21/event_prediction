

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


def kelly_strategy(bet, kelly_fraction, budget=1, kelly_thr=(0, 1)) -> dict:
    if kelly_thr[0] < kelly_fraction < kelly_thr[1]:
        return {'spent': kelly_fraction * budget,
                'potential_gain': bet * kelly_fraction * budget}
    else:
        return {'spent': 0,
                'potential_gain': 0}

def flat_strategy(bet, budget=1):
    return {'spent': budget,
            'potential_gain': budget * bet}

def prob_margin_strategy(bet, prob_margin, budget, margin_thr=(-1,1)) - dict:
    pass