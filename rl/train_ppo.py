# rl/train_ppo.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from preprocessing.data_loader import load_data, preprocess_data
from rl.environment import NetworkEnv
from config.params import PARAMS

DATA_DIR = "../data"


def main():
    print("Loading data...")
    df = load_data(DATA_DIR)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        df, PARAMS["feature_columns"], PARAMS["label_column"]
    )

    print("Creating environment...")
    env = NetworkEnv(X_train, y_train)

    print("Training PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **{
            k: v
            for k, v in PARAMS.items()
            if k in ["learning_rate", "n_steps", "ent_coef", "clip_range", "n_epochs"]
        }
    )
    model.learn(total_timesteps=PARAMS["num_timesteps"])
    model.save("../ppo_model")


if __name__ == "__main__":
    main()
