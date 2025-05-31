# config/params.py

PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "n_epochs": 10,
    "batch_size": 64,
    "feature_columns": [
        'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes',
        'duration', 'proto', 'service', 'conn_state'
    ],
    "label_column": "label_binary",
    "num_timesteps": 10000,
    "test_size": 0.2,
}