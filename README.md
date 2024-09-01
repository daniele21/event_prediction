# Soccer Event Prediction

## APIs

The APIs works running the api server (main.py)

    
    python3 main.py

- [Update](#update)
- [Training](#training)
- [Strategy](#strategy)
- [Go Live](#Go Live)

### Update

cURL

    curl --location 'localhost:8080/api/update' \
        --header 'Content-Type: application/json' \
        --data '{
            "league_name": "serie_a",
            "windows": [1,3,5]
        }'

Python

    import requests
    import json
    
    url = "localhost:8080/api/update"
    
    payload = json.dumps({
      "league_name": "serie_a",
      "windows": [
        1,
        3,
        5
      ]
    })
    headers = {
      'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)

Given a league_name, it download and/or update the league data from the website **football-data.co.uk**
This data contains all the historical data related to that league.

The payload to provide to the api call are:
- *league_name*: name of the league to download/update data (**config/league.py** contains the names of leagues and the 
pointers to the season link data)


- *windows*: list of int --> the integers represents the events to look backward for each team, i.e. in case of windows = [3],
during the creation of training data, there will be created features that looks what happen the 3 matches ago. 
The default is [1,3,5] so the features created looks the behaviour of the last, last 3 and last 5 matches for each team

Data are stored at **/resources/LEAGUE_NAME/ folder**

### Training

#### Hyperparameter Tuning

cURL

    curl --location 'localhost:8080/api/training/hyperparameter_tuning' \
    --header 'Content-Type: application/json' \
    --data '{
        "exp_name": "test",
        "dataset": {
            "league_name": "serie_a",
            "windows": [1, 3, 5],
            "league_dir": "resources/",
            "drop_last_match_days": 5,
            "drop_first_match_days": 5,
            "last_n_seasons": 13,
            "drop_last_seasons": 0,
            "target_match_days": {
                "start": 8,
                "end": 30
            },
            "test_match_day": 2,
            "preprocessing_version": "match_result_v1"
        },
        "training": {
            "estimator": "LGBMClassifier",
            "param_grid": {
                "num_leaves": [5, 10, 15],
                "max_depth": [5, 10, 15],
                "learning_rate": [0.1],
                "n_estimators": [300],
                "early_stopping_round": [10],
                "deterministic": [true],
                "seed": [2024]
            },
            "scoring": "log_loss"
        }
    }'

Python

    import requests
    import json
    
    url = "localhost:8080/api/training/hyperparameter_tuning"
    
    payload = json.dumps({
      "exp_name": "test",
      "dataset": {
        "league_name": "serie_a",
        "windows": [1,3,5],
        "league_dir": "resources/",
        "drop_last_match_days": 5,
        "drop_first_match_days": 5,
        "last_n_seasons": 13,
        "drop_last_seasons": 0,
        "target_match_days": {
          "start": 8,
          "end": 30
        },
        "test_match_day": 2,
        "preprocessing_version": "match_result_v1"
      },
      "training": {
        "estimator": "LGBMClassifier",
        "param_grid": {
          "num_leaves": [5,10,15],
          "max_depth": [5,10,15],
          "learning_rate": [0.1],
          "n_estimators": [300],
          "early_stopping_round": [10],
          "deterministic": [True],
          "seed": [2024]
        },
        "scoring": "log_loss"
      }
    })
    headers = {
      'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)

This section permits to do the hyperparameter tuning of a specified model. The payload is composed by

- exp_name --> it will be the prefix of the folder where the results and config will be saved


- dataset
  
  - league_name
  - windows
  - league_dir --> directory where loading the source data
  - drop_last_match_days --> last match days to skip in a season (drop_last_match_days = 5  for serie_a [38 match days total per season] means to take **until the match day 33**)
  - drop_first_match_days --> first match days to skip in a season (drop_first_match_days = 5  for serie_a means to take **from the match day 5 on**)
  - last_n_seasons --> take just the N last seasons
  - drop_last_seasons --> drop the N last seasons
  - target_match_days --> define the range of match days to consider for the target (consider that the minimum target_match_day is **drop_first_match_days + test_match_day +1** and the maximum is **max(match_days) - drop_last_match_days**)
  - test_match_day --> define the number of match days considered after the training set, before the target match days
  - preprocessing_version --> define the preprocessing to do before the training of data (core/dataset/preprocessing_function.py)


- training

  - estimator --> LGBMClassifier or RandomForestClassifier
  - param_grid --> the params for the tuning with grid search
  - scoring --> the metric to consider for choosing the best params

The results will be stored at **output/{LEAGUE_NAME}/{EXP_NAME}_{DATE}/**

#### Training



This section permits to train a model, starting from the params to give.
You can take the best params provided from the before tuning.

**If you want to use the best params found at tuning stage, you just need to specify the folder where the tuning saved the results and it reload them from there**

So you can provide:

- 