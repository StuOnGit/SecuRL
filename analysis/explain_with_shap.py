# analysis/explain_with_shap.py

import shap
import numpy as np
from stable_baselines3 import PPO
from rl.environment import NetworkEnv
from preprocessing.data_loader import load_data, preprocess_data
from config.params import PARAMS

def explain_model():
    print("[🔄] Carico dati...")
    df = load_data(PARAMS["data_dir"])
    X_train, X_test, y_train, y_test = preprocess_data(df, PARAMS["feature_columns"], PARAMS["label_column"])

    print("[🧠] Carico modello...")
    model = PPO.load("../ppo_model")
    env = NetworkEnv(X_test, y_test)

    def predict_func(data):
        return [model.predict(obs)[0] for obs in data]

    background = shap.kmeans(X_train, 100)
    explainer = shap.KernelExplainer(predict_func, background)
    shap_values = explainer.shap_values(X_test[:10])

    print("[📊] Visualizzo SHAP...")
    shap.summary_plot(shap_values, X_test[:10], feature_names=PARAMS["feature_columns"])

if __name__ == "__main__":
    explain_model()