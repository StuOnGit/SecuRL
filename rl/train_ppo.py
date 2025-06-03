# rl/train_ppo.py

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from preprocessing.data_loader import load_data, preprocess_data
from rl.environment import NetworkEnv
from config.params import PARAMS


def main():
    print("[🔄] Caricamento dati...")
    df = load_data(PARAMS["data_dir"])

    print("\n[🧮] Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(
        df, PARAMS["feature_columns"], PARAMS["label_column"]
    )
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

    print("\n[🎮] Creazione ambiente...")
    env = NetworkEnv(X_train, y_train)

    print("🔍 Verifica NaN:", np.isnan(X_train).sum())
    print("🔍 Verifica Infiniti:", np.isinf(X_train).sum())

    print("\n[🧠] Addestramento PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=PARAMS["learning_rate"],
        n_steps=PARAMS["n_steps"],
    )
    model.learn(total_timesteps=PARAMS["num_timesteps"])
    model.save("../ppo_model")


if __name__ == "__main__":
    main()
