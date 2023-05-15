configs = {
    "setting":{
        "save_dir": "result",
        "plot": False,
    },
    "portfolio_config": {
        "model": "equal_weight", # "model_option": ["All", "GRU", "LSTM", "RNN", "TCN", "Markowitz", "RiskParity", "EqualWeight"]
        "input_period": 240,
        "walk_forward":{
            "training_period": 1200,
            "validation_period": 240,
            "testing_period": 240,
        },
        "vanilla":{
            "training_pct": 0.6,
            "validation_pct": 0.2,
            "testing_pct": 0.2,
        },
        "asset_pool": "sp500_sector_index",
        "fee": 0,
        "weight_diff_threshold": 0.1
    }, 

    "hyperparameters_config": {
        "lr": 0.001,
        "batch_size": 2048, # only for dataloader
        "epoch": 1024,
        "optimization_target": "sharpe",
        "early_stop": 256,
        "weight_decay": 1e-4
    },

    "model_structure_config": {
        "RNN": {
            "hidden_size": 32,
            "num_layers": 4,
            "dropout": 0.0,
            "bidirectional": False,
            "batch_first" : True
        },
        "LSTM": {
            'hidden_size': 32,
            "num_layers": 4,
            "dropout": 0.0,
            "bidirectional": False,
            "batch_first" : True
        },
        "GRU": {
            'hidden_size': 32,
            "num_layers": 4,
            "dropout": 0.0,
            "bidirectional": False,
            "batch_first" : True
        },
        "TCN": {
            "hidden_size": 32,
            "num_layers": 4,
            "kernel_size": 2,
            "dropout": 0.0,
        },
        "TCN2D": {
            "hidden_size": 32,
            "num_layers": 1,
            "kernel_size": 2,
            "dropout": 0.1,
        },
        "Markowitz": {
        },
        "risk_parity": {
        },
        "equal_weight": {
        }
    }
}