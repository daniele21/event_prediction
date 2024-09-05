import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')

def compute_calibration(df, prob_col, event_label):
    df = df[df['choice'] == event_label]

    df['prob_bin'] = pd.cut(df[prob_col],
                            bins=np.arange(0.0, 1.1, 0.1),
                            include_lowest=True)

    actual_freq = df \
        .groupby('prob_bin') \
        .apply(lambda x: (x['result_1X2'] == event_label).mean())

    mean_pred_prob = df.groupby('prob_bin')[prob_col].mean()

    bin_volumes = df['prob_bin'].value_counts().sort_index()

    # Compute confidence intervals
    ci_lower = []
    ci_upper = []
    z = norm.ppf(0.975)  # 95% confidence level

    for p, n in zip(actual_freq, bin_volumes):
        if n > 0:
            # Standard error for proportion
            se = np.sqrt(p * (1 - p) / n)
            # Confidence interval
            ci_lower.append(p - z * se)
            ci_upper.append(p + z * se)
        else:
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)

    return mean_pred_prob, actual_freq, bin_volumes, ci_lower, ci_upper



def check_calibration(data, prob_col, title='', show=True):

    df = data[['choice', prob_col, 'result_1X2']]

    calibration_1 = compute_calibration(df, prob_col, '1')
    calibration_X = compute_calibration(df, prob_col, 'X')
    calibration_2 = compute_calibration(df, prob_col, '2')

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot calibration curves on the primary y-axis
    ax1.plot(calibration_1[0], calibration_1[1], marker='o', label='Home Win (1) Calibration', c='b')
    ax1.plot(calibration_X[0], calibration_X[1], marker='o', label='Draw (X) Calibration', c='green')
    ax1.plot(calibration_2[0], calibration_2[1], marker='o', label='Away Win (2) Calibration', c='r')
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')

    # Add error bars for confidence intervals
    ax1.errorbar(calibration_1[0], calibration_1[1], yerr=[np.array(calibration_1[1]) - np.array(calibration_1[3]),
                                                           np.array(calibration_1[4]) - np.array(calibration_1[1])],
                 fmt='o', capsize=5, c='b')
    ax1.errorbar(calibration_X[0], calibration_X[1], yerr=[np.array(calibration_X[1]) - np.array(calibration_X[3]),
                                                           np.array(calibration_X[4]) - np.array(calibration_X[1])],
                 fmt='o', capsize=5, c='green')
    ax1.errorbar(calibration_2[0], calibration_2[1], yerr=[np.array(calibration_2[1]) - np.array(calibration_2[3]),
                                                           np.array(calibration_2[4]) - np.array(calibration_2[1])],
                 fmt='o', capsize=5, c='r')

    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Observed Frequency')
    ax1.set_title(f'Calibration Curves with Confidence Intervals | {title}')
    ax1.legend(loc='upper left')

    # Create a second y-axis to plot the bin volumes
    ax2 = ax1.twinx()
    ax2.bar(calibration_1[0], calibration_1[2], width=0.05, alpha=0.1, color='b', label='Volume for Home Win (1)',
            align='center')
    ax2.bar(calibration_X[0], calibration_X[2], width=0.05, alpha=0.1, color='green', label='Volume for Draw (X)',
            align='center')
    ax2.bar(calibration_2[0], calibration_2[2], width=0.05, alpha=0.1, color='r', label='Volume for Away Win (2)',
            align='center')
    ax2.set_ylabel('Bin Volume')

    # Combine legends from both axes
    #lines, labels = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



# fig = plt.figure(figsize=(10, 6))
#
# for x in ["1", "X", "2"]:
#
#     a = test_bet_plays[test_bet_plays['choice'] == x][['result_1X2', 'prob']]
#     a['true'] = (a['result_1X2'] == x).astype(int)
#
#     prob_true_rf, prob_pred_rf = calibration_curve(a['true'], a['prob'], n_bins=10)
#
#     # Plot reliability diagram
#     plt.plot(prob_pred_rf, prob_true_rf, marker='o', label=f'{x} | RandomForest (Uncalibrated)')
#     plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
# plt.xlabel('Mean Predicted Probability')
# plt.ylabel('Fraction of Positives')
#
# plt.title('Calibration Curve (Reliability Diagram)')
# plt.legend()
# plt.show()
# fig.savefig('calibration_check.png')