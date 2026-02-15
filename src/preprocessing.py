# src/preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(by=['location_id', 'timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    return df


def preprocess_and_save(
    raw_path,
    processed_folder,
    lookback=24,
    train_ratio=0.8
):
    """
    Full advanced preprocessing pipeline.
    Saves processed numpy arrays into data/processed folder.
    """

    df = load_and_prepare_data(raw_path)

    # -----------------------------
    # One-Hot Encode location_id
    # -----------------------------
    location_dummies = pd.get_dummies(df['location_id'], prefix='loc')
    df = pd.concat([df, location_dummies], axis=1)

    # -----------------------------
    # Scale continuous features
    # -----------------------------
    continuous_features = ['activity_count', 'hour', 'day_of_week']
    scaler = MinMaxScaler()

    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    feature_cols = continuous_features + list(location_dummies.columns)
    target_index = feature_cols.index('activity_count')

    X_train, X_test = [], []
    y_train, y_test = [], []

    # -----------------------------
    # Create sequences per location
    # -----------------------------
    for loc in df['location_id'].unique():

        loc_df = df[df['location_id'] == loc].copy()
        loc_df = loc_df.sort_values(by='timestamp')

        data = loc_df[feature_cols].values

        X_loc, y_loc = [], []

        for i in range(len(data) - lookback):
            X_loc.append(data[i:i+lookback])
            y_loc.append(data[i+lookback][target_index])

        split = int(len(X_loc) * train_ratio)

        X_train.extend(X_loc[:split])
        X_test.extend(X_loc[split:])
        y_train.extend(y_loc[:split])
        y_test.extend(y_loc[split:])

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)


    # -----------------------------
    # Save to processed folder
    # -----------------------------
    os.makedirs(processed_folder, exist_ok=True)

    np.save(os.path.join(processed_folder, "X_train.npy"), X_train)
    np.save(os.path.join(processed_folder, "X_test.npy"), X_test)
    np.save(os.path.join(processed_folder, "y_train.npy"), y_train)
    np.save(os.path.join(processed_folder, "y_test.npy"), y_test)

    print("✓ Processed data saved successfully!")
    print("Saved to:", processed_folder)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return scaler, feature_cols

if __name__ == "__main__":

    RAW_PATH = "./data/raw/simulated_traffic_data.csv"
    PROCESSED_FOLDER = "./data/processed"

    preprocess_and_save(
        raw_path=RAW_PATH,
        processed_folder=PROCESSED_FOLDER,
        lookback=24,
        train_ratio=0.8
    )
