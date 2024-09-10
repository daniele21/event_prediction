import numpy as np


def calculate_implied_probability(odds):
    """
    Calcola la probabilità implicita data una quota.
    """
    return 1 / odds


def calculate_expected_value(prob_estimate, odds):
    """
    Calcola il valore atteso (EV) di una scommessa.

    Arguments:
    prob_estimate -- La probabilità stimata dell'evento (tra 0 e 1).
    odds -- La quota offerta dal bookmaker.

    Returns:
    EV -- Il valore atteso della scommessa.
    """
    return (prob_estimate * odds) - (1 - prob_estimate)


def find_value_bets(prob_estimates, odds):
    """
    Identifica le scommesse di valore (EV positivo).

    Arguments:
    prob_estimates -- Lista o array delle probabilità stimate per ciascun evento.
    odds -- Lista o array delle quote offerte dal bookmaker corrispondenti agli eventi.

    Returns:
    value_bets -- Lista di tuple con (indice dell'evento, EV) per le scommesse con EV positivo.
    """
    value_bets = []
    for i in range(len(prob_estimates)):
        ev = calculate_expected_value(prob_estimates[i], odds[i])

        value_bets.append((i, ev))
    return value_bets


def kelly_criterion(prob_estimate, odds):
    """
    Calcola la frazione di capitale da scommettere usando il criterio di Kelly.

    Arguments:
    prob_estimate -- La probabilità stimata dell'evento (tra 0 e 1).
    odds -- La quota offerta dal bookmaker.

    Returns:
    fraction -- La frazione del bankroll da scommettere (compresa tra 0 e 1).
    """
    kelly_fraction = (prob_estimate * (odds - 1) - (1 - prob_estimate)) / (odds - 1)

    # Se la frazione è negativa, non scommettere (frazione = 0)
    if kelly_fraction < 0:
        return 0
    # Limita la frazione al 100% del bankroll (1)
    return min(kelly_fraction, 1)


def manage_bankroll(fraction, bankroll):
    """
    Gestisce il bankroll calcolando l'ammontare da scommettere.

    Arguments:
    fraction -- La frazione del bankroll da scommettere, ottenuta dal criterio di Kelly.
    bankroll -- Il capitale totale disponibile per le scommesse.

    Returns:
    stake -- L'ammontare da scommettere.
    """
    return fraction * bankroll


def calculate_overround(odds):
    """
    Calcola l'overround di un set di quote.

    Arguments:
    odds -- Lista o array delle quote offerte dal bookmaker.

    Returns:
    overround -- L'overround calcolato.
    """
    implied_probabilities = np.array([calculate_implied_probability(odd) for odd in odds])
    return implied_probabilities.sum() - 1


def arbitrage_opportunity(odds):
    """
    Determina se esiste un'opportunità di arbitraggio.

    Arguments:
    odds -- Lista o array delle migliori quote per ciascun risultato da diversi bookmaker.

    Returns:
    boolean -- True se esiste un'opportunità di arbitraggio, False altrimenti.
    """
    total_implied_prob = sum([calculate_implied_probability(odd) for odd in odds])
    return total_implied_prob < 1





