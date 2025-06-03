# analysis/evaluate.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from stable_baselines3 import PPO
from rl.environment import NetworkEnv
from preprocessing.data_loader import load_data, preprocess_data
from config.params import PARAMS


def main():
    print("\n[🔄] Caricamento dataset...")
    df = load_data(PARAMS["data_dir"])

    print("Label distribution in full dataset:")
    print(Counter(df[PARAMS["label_column"]]))

    print("\n[🧮] Preprocessing...")
    _ , X_test, y_train, y_test = preprocess_data(
        df, PARAMS["feature_columns"], PARAMS["label_column"]
    )

    print("Label distribution after preprocessing:")
    train_counter = Counter(y_train)
    test_counter = Counter(y_test)

    print(f"Train: {train_counter}")
    print(f"Test:  {test_counter}")
    print(f"Total 1s: {train_counter[1]} + {test_counter[1]} = {train_counter[1] + test_counter[1]}")
    print(f"Total 0s: {train_counter[0]} + {test_counter[0]} = {train_counter[0] + test_counter[0]}")

    print("\n[🧠] Carico modello addestrato...")
    try:
        model = PPO.load("../ppo_model")
    except Exception as e:
        print(f"\n[❌] Errore nel caricare il modello: {e}")
        return

    print("\n[🧪] Valutazione modello...")
    env = NetworkEnv(X_test, y_test)
    obs = env.reset()
    print(f"\n[🔍] Primo stato: {obs}")
    print(f"\n[📊] Dimensione osservazione: {obs.shape}")

    predictions = []
    true_labels = []

    for _ in range(len(X_test)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        predictions.append(action)
        true_labels.append(env.labels[env.current_step - 1])
        if done:
            break

    print("\n[📊] METRICHE DI VALUTAZIONE:")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, predictions):.4f}")
    print(f"Recall: {recall_score(true_labels, predictions):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")

    print("\n[📋] CLASSIFICATION REPORT:")
    print(classification_report(true_labels, predictions, digits=4))

    print("\n[🔢] CONFUSION MATRIX:")
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
    # plt.savefig("confusion_matrix.png")
    plt.show()

    df_results = pd.DataFrame({
        "true_label": true_labels,
        "predicted": predictions
    })
    df_results.to_csv("evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()
