from skopt import gp_minimize


def objective(params):
    probability_margin, kelly_fraction_threshold = params
    # Calculate the total profit using the same logic as above
    total_profit = 0
    for match_probs, actual, odd in zip(probabilities, actuals, odds):
        pred_outcome = np.argmax(match_probs)
        p = match_probs[pred_outcome]
        b = odd[pred_outcome] - 1

        if p > probability_margin:
            kelly_fraction = (b * p - (1 - p)) / b
            bet_amount = bankroll * kelly_fraction * kelly_fraction_threshold

            if bet_amount > 0:
                if pred_outcome == actual:
                    total_profit += bet_amount * b
                else:
                    total_profit -= bet_amount
    return -total_profit


# Define the parameter space
param_space = [(0.0, 1.0), (0.0, 1.0)]

# Perform Bayesian optimization
result = gp_minimize(objective, param_space, n_calls=50, random_state=0)

# Extract the optimal parameters
optimal_probability_margin, optimal_kelly_fraction_threshold = result.x
optimal_profit = -result.fun

print(f"Optimal Probability Margin: {optimal_probability_margin}")
print(f"Optimal Kelly Fraction Threshold: {optimal_kelly_fraction_threshold}")
print(f"Optimal Profit: {optimal_profit}")
