# preprocessing/data_loader.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv") and not file.startswith("part-"):
                full_path = os.path.join(root, file)
                try:
                    temp_df = pd.read_csv(full_path, nrows=1)
                    all_files.append(full_path)
                except Exception as e:
                    print(f"\n[⚠️] File non valido: {full_path} ({e})")

    if not all_files:
        raise FileNotFoundError(f"Nessun file CSV valido trovato in '{data_dir}'.")

    dfs = [pd.read_csv(f) for f in all_files]
    return pd.concat(dfs, ignore_index=True)


def calculate_derived_features(df):
    df["pktAtsec"] = df["orig_pkts"] / df["duration"].replace(0, 1)
    df["BitRate"] = (
        (df["orig_bytes"] + df["resp_bytes"]) * 8 / df["duration"].replace(0, 1)
    )
    df["interTime"] = df["duration"] / df["orig_pkts"].replace(0, 1)
    df["avgLenPkt"] = df["orig_bytes"] / df["orig_pkts"].replace(0, 1)
    return df


def preprocess_data(df, feature_cols, label_col="label_binary"):
    # Mappa label_binary
    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "1": 1, "false": 0, "0": 0, "duplicate": 1})
    )

    # Rimuove righe con label mancanti
    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(int)

    # Calcola feature derivate
    df = calculate_derived_features(df)

    continuous_features = [
        "orig_pkts",
        "resp_pkts",
        "orig_bytes",
        "resp_bytes",
        "duration",
        "pktAtsec",
        "BitRate",
        "interTime",
        "avgLenPkt",
    ]
    categorical_features = ["proto", "service", "conn_state"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), continuous_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X = preprocessor.fit_transform(df[feature_cols])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df[label_col]

    # 🔽 Stampa informazioni sulle feature
    print("\n[📊] Dimensione input dopo preprocessing:", X.shape)

    # 🔽 Mostra quanti encoder ha generato
    # one_hot_encoder = preprocessor.named_transformers_["cat"]
    # categories = one_hot_encoder.categories_
    # for i, cat in enumerate(categorical_features):
    #     print(f"[🧩] {cat} → {len(categories[i])} categorie")

    # feature_names = preprocessor.get_feature_names_out()
    # print(len(feature_names))
    # print(feature_names)

    return train_test_split(X, y, test_size=0.2, random_state=42)
