# preprocessing/data_loader.py

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    
    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def preprocess_data(df, feature_cols, label_col="label_binary"):
    continuous_features = ['orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'duration']
    categorical_features = ['proto', 'service', 'conn_state']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), continuous_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X = preprocessor.fit_transform(df[feature_cols])
    y = df[label_col].apply(lambda x: int(x))  # Ensure labels are integers
    return train_test_split(X, y, test_size=0.2, random_state=42)