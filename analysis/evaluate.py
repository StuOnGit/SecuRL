# analysis/evaluate.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from stable_baselines3 import PPO
from rl.environment import NetworkEnv
from preprocessing.data_loader import load_data, preprocess_data
from config.params import PARAMS
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("[🔄] Caricamento dataset...")
    df = load_data(PARAMS["data_dir"])

    print("[🧮] Preprocessing...")
    _, X_test, _, y_test = preprocess_data(
        df, PARAMS["feature_columns"], PARAMS["label_column"]
    )

    print("[🧠] Carico modello addestrato...")
    try:
        model = PPO.load("../ppo_model")
    except Exception as e:
        print(f"[❌] Errore nel caricare il modello: {e}")
        return

    print("[🧪] Valutazione modello...")
    env = NetworkEnv(X_test, y_test)
    obs = env.reset()
    print(f"[🔍] Primo stato: {obs}")
    print(f"[📊] Dimensione osservazione: {obs.shape}")

    predictions = []
    true_labels = []

    for _ in range(len(X_test)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        predictions.append(action)
        true_labels.append(env.labels[env.current_step - 1])
        if done:
            break

    print("\n📊 METRICHE DI VALUTAZIONE:")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, predictions):.4f}")
    print(f"Recall: {recall_score(true_labels, predictions):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")

    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(true_labels, predictions))

    print("\n🔢 CONFUSION MATRIX:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigno", "Malevolo"],
        yticklabels=["Benigno", "Malevolo"],
    )
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.show()


if __name__ == "__main__":
    main()
