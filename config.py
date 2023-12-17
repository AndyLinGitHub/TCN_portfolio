configs = {
    "setting":{
        "save_dir": "result",
        "plot": False,
    },
    "portfolio_config": {
        "model": "TCN",
        "input_length": 240,
        "fee": 0,
    }, 

    "hyperparameters_config": {
        "lr": [0.001],
        "batch_size": [4096],
        "epoch": [256],
        "optimization_target": ["sharpe"],
        "weight_decay": [1e-4]
    },

    "model_structure_configs": {
        "TCN": {
            "num_layers": [4],
            "kernel_size": [16],
            "dropout": [0.2],
        },
        
        "Markowitz": {
        },
        "risk_parity": {
        },
        "equal_weight": {
        }
    }
}