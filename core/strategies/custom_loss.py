import numpy as np

def profit_based_score(y_true, y_pred):
    """
    Custom scoring function to calculate net profit or loss based on a strategy
    that bets only when the predicted net is positive.

    Parameters:
    - y_true: Actual net revenues.
    - y_pred: Predicted net revenues.
    - kellys: Kelly criterion values for each bet (representing bet size).

    Returns:
    - Total profit or loss.
    """
    # Calculate profits only for positive predictions
    # profits = np.where(y_pred > 0, y_true * kellys, 0)
    profits = np.where(y_pred > 0, y_true, 0)

    # Total profit or loss from all bets
    total_profit = np.sum(profits)

    return total_profit


def custom_scorer(estimator, X, y):
    # Predict net using the current estimator
    y_pred = estimator.predict(X)

    # Extract the Kelly values associated with this fold
    kellys_fold = X[:, -1]  # Assuming Kelly values are the last column

    return profit_based_score(y, y_pred, kellys_fold)
