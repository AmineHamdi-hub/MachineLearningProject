# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna()  # Suppression des valeurs manquantes
    return data

def scale_features(data, features):
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[features] = scaler.fit_transform(data[features])
    return data_scaled, scaler
