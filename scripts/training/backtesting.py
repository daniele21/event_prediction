from core.ingestion.update_league_data import get_most_recent_data
from core.simulation.simulation_data import enrich_data_for_simulation
from core.simulation.strategy import extract_margin_matches
from core.strategies.net_strategy import positive_net_strategy
from core.utils import load_pickle
from scripts.model_inference import classification_model_inference


def backtesting_data(data, dataset_params, class_model, net_model):
    prediction, probs = classification_model_inference(data, dataset_params, class_model)

    # Simulation
    simulation_data = enrich_data_for_simulation(data, probs)
    plays = extract_margin_matches(simulation_data)

    _, fig = positive_net_strategy(plays,
                                   net_model,
                                   info_data='target set',
                                   #save_folder=source_folder
                                   )

    fig.savefig('test.png')

    return fig


if __name__ == '__main__':
    dataset_config = {
        "league_name": "serie_a",
        "windows": [1, 3, 5],
        "league_dir": "resources/",
        "drop_last_match_days": 5,
        "drop_first_match_days": 0,
        "last_n_seasons": 13,
        "drop_last_seasons": 0,
        "target_match_days": {"start": 1, "end": 3},
        "test_match_day": 0,
        "preprocessing_version": "match_result_v1"
    }

    source_data = get_most_recent_data(dataset_config)
    source_folder = 'outputs/e2e/serie_a/all_seasons_e2e_8_28_2223_20240901_110500'

    class_model_path = f'{source_folder}/class_model.pkl'
    net_model_path = f'{source_folder}/net_prediction_model.pkl'
    class_model = load_pickle(class_model_path)
    net_model = load_pickle(net_model_path)


    backtesting_data(source_data, dataset_config, class_model, net_model)