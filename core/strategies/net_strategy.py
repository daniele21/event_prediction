from sklearn.metrics import classification_report

from core.logger import logger
from scripts.strategy import plot_profit_loss
import pandas as pd


def positive_net_strategy(data, net_model, info_data='', save_folder=None,
                          budget=100, inference=False):

    df = predict_net(data, net_model)

    df['bet_decision'] = df['predicted_net'].apply(lambda x: 1 if x > 0 else 0)
    df = df.sort_values(by='match_day')

    # budget = 1
    # dynamic_budget = pd.DataFrame()
    # for match_day, group in df.groupby('match_day'):
    #     budget_dict = {'budget': budget}
    #     day_budget = budget / len(group)
    #     day_net = (day_budget * group['net']).sum()
    #     budget += day_net
    #     budget_dict.update(**{'match_day': match_day,
    #                            'updated_budget': budget,
    #                            'match_budget': day_budget,
    #                            'day_net': day_net})
    #     dynamic_budget = pd.concat((dynamic_budget,
    #                                 pd.DataFrame(budget_dict, index=[match_day])))

    if not inference:

        class_report = classification_report(df['win'].astype(int),
                                             df['bet_decision'],
                                             output_dict=True)
        print(f'Bet Decision Classification Report | Test')
        print(classification_report(df['win'].astype(int),
                                    df['bet_decision']))

        fig = plot_profit_loss(df[df['bet_decision'] == 1], show=False)

        if save_folder:
            pl_path = f'{save_folder}/positive_net_strategy_on_{info_data}'
            logger.info(f' > Saving Net/Spent on {info_data} at {pl_path}')
            fig.savefig(pl_path)

            bet_path = f'{save_folder}/bet_positive_net_decision_{info_data}.csv'
            logger.info(f' > Saving bet decision {info_data} at {bet_path}')
            df.to_csv(bet_path)

            class_report_path = f'{save_folder}/bet_positive_net_decision_class_report_{info_data}.csv'
            logger.info(f' > Saving bet decision classification report {info_data} at {class_report_path}')
            pd.DataFrame(class_report).T.to_csv(class_report_path)

        return df, fig

    return df, None


def predict_net(data, net_model):
    features = ['ev', 'kelly', 'prob_margin']

    df = data[data['kelly'] > 0][features].drop_duplicates()
    net_prediction = net_model.predict(df)

    df.loc[:, 'predicted_net'] = net_prediction
    df = data.drop(features, axis=1).merge(df, how='right', left_index=True, right_index=True)

    return df