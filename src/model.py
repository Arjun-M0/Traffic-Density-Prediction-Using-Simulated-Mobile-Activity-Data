# src/model.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras import layers, models
import json

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


def load_processed_data(processed_folder):
    """Load preprocessed numpy arrays from data/processed folder."""
    X_train = np.load(os.path.join(processed_folder, "X_train.npy"))
    X_test = np.load(os.path.join(processed_folder, "X_test.npy"))
    y_train = np.load(os.path.join(processed_folder, "y_train.npy"))
    y_test = np.load(os.path.join(processed_folder, "y_test.npy"))
    
    return X_train, X_test, y_train, y_test


def make_sequences(data, lookback=24):
    """
    Convert raw data into sequences for model inference.
    
    Args:
        data (np.ndarray): Input features array of shape (num_samples, num_features)
        lookback (int): Number of timesteps to look back. Default is 24.
    
    Returns:
        np.ndarray: Sequences of shape (num_sequences, lookback, num_features)
    """
    sequences = []
    
    for i in range(len(data) - lookback):
        sequences.append(data[i:i+lookback])
    
    return np.array(sequences, dtype=np.float32)


def build_lstm_model(input_shape):
    """Build LSTM-based sequence prediction model."""
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the model with early stopping and learning rate scheduling."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test, verbose=0).ravel()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE: only calculate for non-zero actual values to avoid division issues
    mask = y_test > 0.1  # Filter out near-zero values
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = 0.0
    
    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape)
    }
    
    return y_pred, metrics


def plot_training_history(history, save_path):
    """Plot training and validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Model MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"✓ Training history plot saved to: {save_path}")
    plt.close()


def plot_predictions(y_test, y_pred, save_path):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 5))
    
    plt.plot(y_test[:200], label='Actual', marker='o', markersize=3, alpha=0.7)
    plt.plot(y_pred[:200], label='Predicted', marker='s', markersize=3, alpha=0.7)
    
    plt.xlabel('Test Sample Index')
    plt.ylabel('Traffic Density (Normalized)')
    plt.title('Actual vs Predicted Traffic Density (First 200 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"✓ Predictions plot saved to: {save_path}")
    plt.close()


def main():
    # Paths
    processed_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_folder, exist_ok=True)
    
    print("=" * 60)
    print("Traffic Density Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_processed_data(processed_folder)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test shape: {y_test.shape}")
    
    # Build model
    print("\n2. Building LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    print("   Model architecture:")
    model.summary()
    
    # Train model
    print("\n3. Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    
    print("\n   Model Performance:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    # Save results
    print("\n5. Saving results...")
    
    # Save model
    model_path = os.path.join(results_folder, 'traffic_model.h5')
    model.save(model_path)
    print(f"   ✓ Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(results_folder, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"   ✓ Metrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_path = os.path.join(results_folder, 'predictions.npy')
    np.save(predictions_path, y_pred.astype(np.float32))
    print(f"   ✓ Predictions saved to: {predictions_path}")
    
    # Save actual values
    actual_path = os.path.join(results_folder, 'y_test_actual.npy')
    np.save(actual_path, y_test)
    print(f"   ✓ Actual test values saved to: {actual_path}")
    
    # Generate plots
    print("\n6. Generating plots...")
    plot_training_history(history, os.path.join(results_folder, 'training_history.png'))
    plot_predictions(y_test, y_pred, os.path.join(results_folder, 'predictions_plot.png'))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()