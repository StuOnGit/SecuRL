# analysis/evaluate.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stable_baselines3 import PPO
from rl.environment import NetworkEnv
from preprocessing.data_loader import load_data, preprocess_data
from config.params import PARAMS

DATA_DIR = "../data"

def main():
    print("Loading data...")
    df = load_data(DATA_DIR)

    print("Preprocessing data...")
    _, X_test, _, y_test = preprocess_data(df, PARAMS["feature_columns"], PARAMS["label_column"])

    print("Loading trained model...")
    model = PPO.load("../ppo_model")

    print("Evaluating model...")
    env = NetworkEnv(X_test, y_test)
    obs = env.reset()

    actions = []
    true_labels = []

    for _ in range(len(X_test)):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, done, _ = env.step(action)
        true_labels.append(env.labels[env.current_step - 1])
        if done:
            break

    print("Metrics:")
    print(f"Accuracy: {accuracy_score(true_labels, actions):.4f}")
    print(f"Precision: {precision_score(true_labels, actions):.4f}")
    print(f"Recall: {recall_score(true_labels, actions):.4f}")
    print(f"F1 Score: {f1_score(true_labels, actions):.4f}")

if __name__ == "__main__":
    main()