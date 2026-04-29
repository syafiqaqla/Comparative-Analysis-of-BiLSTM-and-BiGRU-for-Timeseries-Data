"""
Stock Prediction Utilities
==========================
Shared utility module for "A Comparative Analysis of BiLSTM and BiGRU 
for Stock Price Prediction" research pipeline.

Author: Research Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
PROPORTION_SCALER_MAX = 10501.0 # BBCA ATH
LOOKBACK = 1
EPOCHS = 100
BATCH_SIZE = 64
UNITS = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
RANDOM_SEED = 42
PATIENCE = 20  # Early stopping patience

STOCKS = ['TLKM', 'BBCA', 'ASII', 'UNVR']
MODEL_TYPES = ['BiLSTM', 'BiGRU', 'LSTM', 'GRU']
TIMEFRAMES = ['weekly', 'monthly', 'yearly']

# Color palettes (colorblind-friendly, distinct)
MODEL_COLORS = {
    'BiLSTM': '#0072B2',   # Blue
    'BiGRU': '#D55E00',    # Vermillion/Orange
    'LSTM': '#009E73',     # Bluish Green
    'GRU': '#CC79A7',      # Reddish Purple
}
STOCK_COLORS = {
    'TLKM': '#0072B2',
    'BBCA': '#D55E00',
    'ASII': '#009E73',
    'UNVR': '#CC79A7',
}
ACTUAL_COLOR = "#9A9A9A"  # Black for actual values

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================================================
# GPU SETUP
# ============================================================
def setup_gpu():
    """
    Configure TensorFlow to use GPU (CUDA) if available.
    Falls back to CPU if GPU is not available.
    """
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n{'='*60}")
        print(f"GPU SETUP - CUDA Available")
        print(f"{'='*60}")
        print(f"Number of GPUs detected: {len(gpus)}")
        
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\nMemory growth enabled (dynamic allocation)")
            
            # Set TensorFlow to use GPU by default
            tf.config.set_visible_devices(gpus, 'GPU')
            print("TensorFlow configured to use GPU")
            print(f"{'='*60}\n")
            
            return True
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
            print("Falling back to CPU")
            return False
    else:
        print(f"\n{'='*60}")
        print("GPU SETUP - No GPU/CUDA Detected")
        print(f"{'='*60}")
        print("Training will use CPU (slower)")
        print("If you have an NVIDIA GPU, make sure CUDA and cuDNN are installed")
        print(f"{'='*60}\n")
        return False

def check_gpu():
    """Check and display current GPU/CPU status."""
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"\nDevice Configuration:")
    print(f"  GPUs available: {len(gpus)}")
    print(f"  CPUs available: {len(cpus)}")
    
    if gpus:
        print(f"  TensorFlow will use GPU for computations")
    else:
        print(f"  TensorFlow will use CPU for computations (no GPU found)")

# ============================================================
# IEEE PLOT STYLE
# ============================================================
def set_ieee_style():
    """Configure matplotlib for IEEE publication-quality figures."""
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'legend.framealpha': 0.9,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_palette("colorblind")

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def load_stock_data(filepath):
    """Load stock data from Excel file. Returns DataFrame with Date and Close columns."""
    df = pd.read_excel(filepath)
    # Keep only Date and Close, drop NaN
    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def get_data_paths(data_dir, stock, timeframe='daily'):
    """Get file path for a given stock and timeframe."""
    if timeframe == 'daily':
        return os.path.join(data_dir, f'{stock.lower()}_daily_data.xlsx')
    else:
        return os.path.join(data_dir, f'{stock.lower()}_{timeframe}_data.xlsx')

def load_all_daily_data(data_dir):
    """Load daily data for all stocks. Returns dict of DataFrames."""
    data = {}
    for stock in STOCKS:
        filepath = get_data_paths(data_dir, stock, 'daily')
        data[stock] = load_stock_data(filepath)
        print(f"  {stock}: {len(data[stock])} records, "
              f"Date range: {data[stock]['Date'].min().date()} to {data[stock]['Date'].max().date()}")
    return data

# ============================================================
# PROPORTION SCALER (No data leakage - fixed constant)
# ============================================================
def proportion_scale(data, max_val=PROPORTION_SCALER_MAX):
    """Scale data by dividing by BBCA ATH (fixed constant = 10501)."""
    return data / max_val

def proportion_inverse_scale(data, max_val=PROPORTION_SCALER_MAX):
    """Inverse scale data by multiplying by BBCA ATH."""
    return data * max_val

# ============================================================
# SEQUENCE CREATION
# ============================================================
def create_sequences(data, lookback=LOOKBACK):
    """
    Create input-output sequences for time series prediction.
    
    Args:
        data: 1D numpy array of scaled values
        lookback: number of past timesteps to use as input
    
    Returns:
        X: shape (n_samples, lookback, 1)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    # Reshape X to (samples, lookback, 1) for RNN input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def prepare_same_stock_data(df, train_ratio=0.8, lookback=LOOKBACK):
    """
    Prepare data for same-stock prediction (Experiment 1).
    Data-leakage-proof: chronological split before sequence creation.
    
    Returns:
        X_train, y_train, X_test, y_test (all in scaled space)
        test_dates: dates corresponding to y_test
    """
    close_values = df['Close'].values.astype(np.float64)
    dates = df['Date'].values
    
    # Scale using ProportionScaler (fixed constant, no leakage)
    scaled = proportion_scale(close_values)
    
    # Chronological split
    split_idx = int(len(scaled) * train_ratio)
    
    # Training data: create sequences from training portion only
    train_data = scaled[:split_idx]
    X_train, y_train = create_sequences(train_data, lookback)
    
    # Test data: use last 'lookback' points from training + test portion
    # This ensures the first test sample has a proper lookback window
    test_data = scaled[split_idx - lookback:]
    X_test, y_test = create_sequences(test_data, lookback)
    
    # Get corresponding dates for test predictions
    test_dates = dates[split_idx:]
    
    return X_train, y_train, X_test, y_test, test_dates

def prepare_cross_stock_data(train_df, test_df, train_ratio=0.8, lookback=LOOKBACK):
    """
    Prepare data for cross-stock prediction (Experiment 2).
    Train on one stock, predict on another stock's test portion.
    
    Returns:
        X_train, y_train: from train_stock's training portion
        X_test, y_test: from test_stock's test portion
        test_dates: dates for test predictions
    """
    # Training stock
    train_close = proportion_scale(train_df['Close'].values.astype(np.float64))
    split_train = int(len(train_close) * train_ratio)
    train_data = train_close[:split_train]
    X_train, y_train = create_sequences(train_data, lookback)
    
    # Test stock - use its test portion
    test_close = proportion_scale(test_df['Close'].values.astype(np.float64))
    test_dates_all = test_df['Date'].values
    split_test = int(len(test_close) * train_ratio)
    test_data = test_close[split_test - lookback:]
    X_test, y_test = create_sequences(test_data, lookback)
    test_dates = test_dates_all[split_test:]
    
    return X_train, y_train, X_test, y_test, test_dates

def prepare_diff_timeframe_data(train_df, test_daily_df, train_ratio=0.8, lookback=LOOKBACK):
    """
    Prepare data for different-timeframe prediction (Experiment 3).
    Train on weekly/monthly/yearly data, predict on daily test data.
    
    Data leakage prevention: only use training timeframe data 
    up to the daily split date.
    """
    # Determine the split date from daily data
    daily_dates = test_daily_df['Date'].values
    split_idx_daily = int(len(daily_dates) * train_ratio)
    split_date = daily_dates[split_idx_daily]
    
    # Training data: use only timeframe data up to split date
    train_mask = train_df['Date'].values <= split_date
    train_portion = train_df[train_mask]
    
    if len(train_portion) < lookback + 1:
        print(f"  WARNING: Not enough training data ({len(train_portion)} records). "
              f"Need at least {lookback + 1}.")
        return None, None, None, None, None
    
    train_close = proportion_scale(train_portion['Close'].values.astype(np.float64))
    X_train, y_train = create_sequences(train_close, lookback)
    
    # Test data: daily test portion
    daily_close = proportion_scale(test_daily_df['Close'].values.astype(np.float64))
    test_data = daily_close[split_idx_daily - lookback:]
    X_test, y_test = create_sequences(test_data, lookback)
    test_dates = daily_dates[split_idx_daily:]
    
    return X_train, y_train, X_test, y_test, test_dates

def prepare_multi_stock_data(train_dfs, test_df, train_ratio=0.8, lookback=LOOKBACK):
    """
    Prepare data for multi-stock training (Experiment 4).
    Train on multiple stocks' training portions combined, predict on target test.
    
    Args:
        train_dfs: list of DataFrames for training stocks
        test_df: DataFrame for target stock
    """
    # Combine training sequences from all training stocks
    all_X_train, all_y_train = [], []
    for df in train_dfs:
        close = proportion_scale(df['Close'].values.astype(np.float64))
        split_idx = int(len(close) * train_ratio)
        train_data = close[:split_idx]
        X, y = create_sequences(train_data, lookback)
        all_X_train.append(X)
        all_y_train.append(y)
    
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    
    # Shuffle training data (sequences are independent, shuffling is ok)
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    # Test data: target stock's test portion
    test_close = proportion_scale(test_df['Close'].values.astype(np.float64))
    test_dates_all = test_df['Date'].values
    split_test = int(len(test_close) * train_ratio)
    test_data = test_close[split_test - lookback:]
    X_test, y_test = create_sequences(test_data, lookback)
    test_dates = test_dates_all[split_test:]
    
    return X_train, y_train, X_test, y_test, test_dates

# ============================================================
# MODEL BUILDING
# ============================================================
def build_model(model_type, lookback=LOOKBACK, units=UNITS, dropout=DROPOUT_RATE):
    """
    Build a sequential RNN model.
    
    Args:
        model_type: one of 'BiLSTM', 'BiGRU', 'LSTM', 'GRU'
        lookback: input sequence length
        units: number of hidden units per layer
        dropout: dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential(name=model_type)
    
    if model_type == 'BiLSTM':
        model.add(Input(shape=(lookback, 1)))
        model.add(Bidirectional(LSTM(units, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(units, return_sequences=False)))
        model.add(Dropout(dropout))
    elif model_type == 'BiGRU':
        model.add(Input(shape=(lookback, 1)))
        model.add(Bidirectional(GRU(units, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(GRU(units, return_sequences=False)))
        model.add(Dropout(dropout))
    elif model_type == 'LSTM':
        model.add(Input(shape=(lookback, 1)))
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(dropout))
    elif model_type == 'GRU':
        model.add(Input(shape=(lookback, 1)))
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(units, return_sequences=False))
        model.add(Dropout(dropout))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    
    return model

# ============================================================
# EVALUATION METRICS
# ============================================================
def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics on original (inverse-scaled) values.
    
    Returns:
        dict with MSE, RMSE, MAE, MAPE, R2
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE: avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE (%)': round(mape, 4),
        'R2': round(r2, 6)
    }

# ============================================================
# TRAINING PIPELINE
# ============================================================
def train_and_evaluate(model_type, X_train, y_train, X_test, y_test,
                       experiment_name, save_dir='models',
                       epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Full training and evaluation pipeline.
    
    Args:
        model_type: 'BiLSTM', 'BiGRU', 'LSTM', 'GRU'
        X_train, y_train: training data (scaled)
        X_test, y_test: test data (scaled)
        experiment_name: string for file naming
        save_dir: directory to save models
        
    Returns:
        y_true_inv: actual values (original scale)
        y_pred_inv: predicted values (original scale)
        metrics: dict of evaluation metrics
        history: training history
    """
    set_seed()
    os.makedirs(save_dir, exist_ok=True)
    
    # Build model
    model = build_model(model_type)
    
    # File paths
    safe_name = experiment_name.replace(' ', '_').replace('/', '_')
    best_path = os.path.join(save_dir, f'{safe_name}_{model_type}_best.keras')
    last_path = os.path.join(save_dir, f'{safe_name}_{model_type}_last.keras')
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=best_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"\n{'='*60}")
    print(f"Training {model_type} for: {experiment_name}")
    print(f"  Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Train (validation_split takes last 10% of training data - chronologically correct)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint], # Early stopping will restore best weights, so no need to save last model separately
        shuffle=True,  # Shuffling independent sequences is fine
        verbose=1
    )
    
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.1f}s ({len(history.history['loss'])} epochs)")
    
    # Save last model
    model.save(last_path)
    print(f"  Best model saved: {best_path}")
    print(f"  Last model saved: {last_path}")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Inverse scale to original values
    y_true_inv = proportion_inverse_scale(y_test)
    y_pred_inv = proportion_inverse_scale(y_pred_scaled)
    
    # Evaluate
    metrics = evaluate_predictions(y_true_inv, y_pred_inv)
    metrics['Training_Time_s'] = round(train_time, 1)
    metrics['Epochs_Run'] = len(history.history['loss'])
    
    print(f"  Results: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
          f"MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE (%)']:.2f}%, R2={metrics['R2']:.6f}")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    return y_true_inv, y_pred_inv, metrics, history

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def save_fig(fig, filepath, dpi=600):
    """Save figure at publication quality."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Figure saved: {filepath}")

def plot_actual_vs_predicted(test_dates, y_true, y_pred, model_type, stock,
                              experiment_label, save_dir='figures'):
    """Plot actual vs predicted stock prices."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, y_true, color=ACTUAL_COLOR, label='Actual', linewidth=1.5)
    ax.plot(test_dates, y_pred, color=MODEL_COLORS[model_type], 
            label=f'Predicted ({model_type})', linewidth=1.5, linestyle='--')
    
    ax.set_title(f'{experiment_label}\n{model_type} - {stock} Stock Price Prediction', fontsize=14)
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Close Price (IDR)', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    
    fname = f'{save_dir}/{experiment_label}_{stock}_{model_type}_prediction.png'.replace(' ', '_')
    save_fig(fig, fname)

def plot_all_models_comparison(test_dates, y_true, predictions_dict, stock,
                                experiment_label, save_dir='figures'):
    """
    Plot actual vs all 4 model predictions on one chart.
    predictions_dict: {model_type: y_pred_array}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(test_dates, y_true, color=ACTUAL_COLOR, label='Actual', 
            linewidth=2.0, zorder=5)
    
    linestyles = {'BiLSTM': '--', 'BiGRU': '-.', 'LSTM': ':', 'GRU': (0, (3, 1, 1, 1))}
    
    for model_type in MODEL_TYPES:
        if model_type in predictions_dict:
            ax.plot(test_dates, predictions_dict[model_type],
                    color=MODEL_COLORS[model_type],
                    label=model_type,
                    linewidth=1.5,
                    linestyle=linestyles.get(model_type, '--'),
                    alpha=0.85)
    
    ax.set_title(f'{experiment_label}\nAll Models Comparison - {stock}', fontsize=14)
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Close Price (IDR)', fontsize=13)
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    
    fname = f'{save_dir}/{experiment_label}_{stock}_all_models.png'.replace(' ', '_')
    save_fig(fig, fname)

def plot_training_history(history, model_type, stock, experiment_label, save_dir='figures'):
    """Plot training and validation loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], label='Training Loss', 
            color=MODEL_COLORS[model_type], linewidth=1.5)
    ax.plot(history.history['val_loss'], label='Validation Loss',
            color=MODEL_COLORS[model_type], linewidth=1.5, linestyle='--')
    
    ax.set_title(f'{experiment_label}\n{model_type} - {stock} Training History', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss (MSE)', fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    
    fname = f'{save_dir}/{experiment_label}_{stock}_{model_type}_history.png'.replace(' ', '_')
    save_fig(fig, fname)

def plot_metrics_comparison_bar(results_df, metric, experiment_label, 
                                 group_col='Stock', save_dir='figures'):
    """Bar chart comparing a metric across models and stocks."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = results_df[group_col].unique()
    n_groups = len(groups)
    n_models = len(MODEL_TYPES)
    bar_width = 0.8 / n_models
    x = np.arange(n_groups)
    
    for i, model_type in enumerate(MODEL_TYPES):
        model_data = results_df[results_df['Model'] == model_type]
        values = []
        for grp in groups:
            val = model_data[model_data[group_col] == grp][metric]
            values.append(val.values[0] if len(val) > 0 else 0)
        
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=model_type, color=MODEL_COLORS[model_type],
                      edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel(group_col, fontsize=13)
    ax.set_ylabel(metric, fontsize=13)
    ax.set_title(f'{experiment_label}\n{metric} Comparison Across Models', fontsize=14)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(groups, fontsize=11)
    ax.legend(fontsize=11, loc='best')
    fig.tight_layout()
    
    fname = f'{save_dir}/{experiment_label}_{metric}_comparison.png'.replace(' ', '_').replace('(%)', 'pct')
    save_fig(fig, fname)

def plot_metrics_heatmap(results_df, metric, experiment_label,
                          row_col='Train_Stock', col_col='Test_Stock', 
                          save_dir='figures'):
    """Heatmap for cross-stock or multi-experiment results."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_type in MODEL_TYPES:
        model_data = results_df[results_df['Model'] == model_type]
        if model_data.empty:
            continue
        
        pivot = model_data.pivot_table(
            values=metric, index=row_col, columns=col_col, aggfunc='first'
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                    ax=ax, linewidths=0.5, linecolor='white')
        ax.set_title(f'{experiment_label}\n{model_type} - {metric}', fontsize=14)
        ax.set_xlabel(col_col.replace('_', ' '), fontsize=13)
        ax.set_ylabel(row_col.replace('_', ' '), fontsize=13)
        fig.tight_layout()
        
        fname = (f'{save_dir}/{experiment_label}_{model_type}_{metric}_heatmap.png'
                 .replace(' ', '_').replace('(%)', 'pct'))
        save_fig(fig, fname)

# ============================================================
# INTERACTIVE DATA SPLITTING VISUALIZATION (Plotly)
# ============================================================
def create_interactive_data_split_visualization(df, train_ratio, stock_name, 
                                                experiment_label, save_dir='figures'):
    """
    Create an interactive visualization showing the train/test data split.
    
    Args:
        df: DataFrame with 'Date' and 'Close' columns
        train_ratio: Train/test split ratio (e.g., 0.8 for 80/20)
        stock_name: Name of the stock
        experiment_label: Label for the experiment (e.g., 'Exp1_80_20')
        save_dir: Directory to save the HTML file
    
    Returns:
        fig: Plotly figure object
    """
    os.makedirs(save_dir, exist_ok=True)
    
    close_values = df['Close'].values.astype(np.float64)
    # Convert dates to pandas DatetimeIndex for consistent type handling
    dates = pd.to_datetime(df['Date'].values)
    
    # Calculate split point
    split_idx = int(len(close_values) * train_ratio)
    split_date = dates[split_idx]  # Already a Timestamp from DatetimeIndex
    
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    train_prices = close_values[:split_idx]
    test_prices = close_values[split_idx:]
    
    # Create figure with secondary y-axis for visualization
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(
        go.Scatter(
            x=train_dates, y=train_prices,
            name='Training Data',
            mode='lines',
            line=dict(color='#0072B2', width=2.5),
            hovertemplate='<b>Training</b><br>Date: %{x|%Y-%m-%d}<br>Price: IDR %{y:,.2f}<extra></extra>',
        )
    )
    
    # Add test data
    fig.add_trace(
        go.Scatter(
            x=test_dates, y=test_prices,
            name='Test Data',
            mode='lines',
            line=dict(color='#D55E00', width=2.5),
            hovertemplate='<b>Test</b><br>Date: %{x|%Y-%m-%d}<br>Price: IDR %{y:,.2f}<extra></extra>',
        )
    )
    
    # Add split point marker (without annotation to avoid Plotly datetime calculation issues)
    fig.add_vline(
        x=split_date,
        line_dash="dash",
        line_color="red",
        opacity=0.7
    )
    
    # Add annotation for split point separately
    fig.add_annotation(
        text=f"<b>Split Point</b><br>{split_date.strftime('%Y-%m-%d')}",
        x=split_date,
        y=1.0,
        yref="paper",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=0,
        ay=-50,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="red",
        borderwidth=1,
        font=dict(size=10, color="red", family="Times New Roman")
    )
    
    # Calculate statistics
    train_count = len(train_prices)
    test_count = len(test_prices)
    total_count = train_count + test_count
    train_pct = (train_count / total_count) * 100
    test_pct = (test_count / total_count) * 100
    
    # Create statistics text
    stats_text = (
        f"<b>Data Split Statistics</b><br>"
        f"Train Samples: {train_count} ({train_pct:.1f}%)<br>"
        f"Test Samples: {test_count} ({test_pct:.1f}%)<br>"
        f"Total Samples: {total_count}<br>"
        f"<br>"
        f"<b>Date Range</b><br>"
        f"Start: {dates[0].strftime('%Y-%m-%d')}<br>"
        f"Split: {split_date.strftime('%Y-%m-%d')}<br>"
        f"End: {dates[-1].strftime('%Y-%m-%d')}<br>"
        f"<br>"
        f"<b>Price Statistics</b><br>"
        f"Train Min: IDR {train_prices.min():,.2f}<br>"
        f"Train Max: IDR {train_prices.max():,.2f}<br>"
        f"Test Min: IDR {test_prices.min():,.2f}<br>"
        f"Test Max: IDR {test_prices.max():,.2f}"
    )
    
    # Add annotation box with statistics
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=11, family="Times New Roman"),
        align="left",
        xanchor="left",
        yanchor="top"
    )
    
    # Update layout
    fig.update_layout(
        title=f"<b>{stock_name} - Data Split Visualization ({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f})</b>",
        xaxis_title="Date",
        yaxis_title="Price (IDR)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        font=dict(size=12, family="Times New Roman"),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type="date",
            gridwidth=1,
            gridcolor="lightgray"
        ),
        yaxis=dict(
            gridwidth=1,
            gridcolor="lightgray"
        ),
        legend=dict(
            x=0.99,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        showlegend=True
    )
    
    # Save as HTML
    ratio_str = f"{int(train_ratio*100)}_{int((1-train_ratio)*100)}"
    html_filename = f'{save_dir}/interactive_data_split_{stock_name}_{ratio_str}.html'
    fig.write_html(html_filename)
    
    return fig, html_filename

def create_interactive_split_summary_visualization(daily_data, train_ratio, 
                                                    experiment_label, 
                                                    save_dir='figures'):
    """
    Create a summary visualization showing data splits for all stocks.
    
    Args:
        daily_data: Dictionary of DataFrames {stock_name: df}
        train_ratio: Train/test split ratio
        experiment_label: Label for experiment
        save_dir: Directory to save
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots for all stocks
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=STOCKS,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    row_col_pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, stock in enumerate(STOCKS):
        row, col = row_col_pairs[idx]
        df = daily_data[stock]
        
        close_values = df['Close'].values.astype(np.float64)
        # Convert dates to pandas DatetimeIndex for consistent type handling
        dates = pd.to_datetime(df['Date'].values)
        
        split_idx = int(len(close_values) * train_ratio)
        split_date = dates[split_idx]  # Already a Timestamp from DatetimeIndex
        
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        train_prices = close_values[:split_idx]
        test_prices = close_values[split_idx:]
        
        # Add training data
        fig.add_trace(
            go.Scatter(
                x=train_dates, y=train_prices,
                name='Training',
                mode='lines',
                line=dict(color='#0072B2', width=1.5),
                hovertemplate='<b>Train</b><br>%{x|%Y-%m-%d}<br>IDR %{y:,.0f}<extra></extra>',
                showlegend=(idx == 0),
                legendgroup="train"
            ),
            row=row, col=col
        )
        
        # Add test data
        fig.add_trace(
            go.Scatter(
                x=test_dates, y=test_prices,
                name='Test',
                mode='lines',
                line=dict(color='#D55E00', width=1.5),
                hovertemplate='<b>Test</b><br>%{x|%Y-%m-%d}<br>IDR %{y:,.0f}<extra></extra>',
                showlegend=(idx == 0),
                legendgroup="test"
            ),
            row=row, col=col
        )
        
        # Add split line
        fig.add_vline(
            x=split_date,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Price (IDR)", row=row, col=col)
    
    # Update layout
    ratio_label = f"{int(train_ratio*100)}/{int((1-train_ratio)*100)}"
    fig.update_layout(
        title=f"<b>Data Split Visualization - All Stocks ({ratio_label})</b>",
        height=900,
        template='plotly_white',
        hovermode='x unified',
        font=dict(size=10, family="Times New Roman"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save as HTML
    ratio_str = f"{int(train_ratio*100)}_{int((1-train_ratio)*100)}"
    html_filename = f'{save_dir}/interactive_data_split_all_stocks_{ratio_str}.html'
    fig.write_html(html_filename)
    
    return fig, html_filename

def create_interactive_split_bar_chart(daily_data, train_ratio, 
                                       experiment_label, save_dir='figures'):
    """
    Create an interactive bar chart showing train/test split statistics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    stats_data = {
        'Stock': [],
        'Training Samples': [],
        'Test Samples': [],
        'Total Samples': [],
        'Train %': [],
        'Test %': []
    }
    
    for stock in STOCKS:
        df = daily_data[stock]
        total = len(df)
        split_idx = int(total * train_ratio)
        train_count = split_idx
        test_count = total - split_idx
        
        stats_data['Stock'].append(stock)
        stats_data['Training Samples'].append(train_count)
        stats_data['Test Samples'].append(test_count)
        stats_data['Total Samples'].append(total)
        stats_data['Train %'].append(train_ratio * 100)
        stats_data['Test %'].append((1 - train_ratio) * 100)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Training',
        x=stats_data['Stock'],
        y=stats_data['Training Samples'],
        marker_color='#0072B2',
        hovertemplate='<b>%{x} - Training</b><br>Samples: %{y}<extra></extra>',
        text=stats_data['Training Samples'],
        textposition='inside',
        textfont=dict(color='white', size=11, family="Times New Roman")
    ))
    
    fig.add_trace(go.Bar(
        name='Test',
        x=stats_data['Stock'],
        y=stats_data['Test Samples'],
        marker_color='#D55E00',
        hovertemplate='<b>%{x} - Test</b><br>Samples: %{y}<extra></extra>',
        text=stats_data['Test Samples'],
        textposition='inside',
        textfont=dict(color='white', size=11, family="Times New Roman")
    ))
    
    ratio_label = f"{int(train_ratio*100)}/{int((1-train_ratio)*100)}"
    fig.update_layout(
        title=f"<b>Data Split Statistics - All Stocks ({ratio_label})</b>",
        barmode='stack',
        xaxis_title="Stock",
        yaxis_title="Sample Count",
        template='plotly_white',
        height=500,
        font=dict(size=12, family="Times New Roman"),
        hovermode='x unified',
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    # Save as HTML
    ratio_str = f"{int(train_ratio*100)}_{int((1-train_ratio)*100)}"
    html_filename = f'{save_dir}/interactive_split_statistics_{ratio_str}.html'
    fig.write_html(html_filename)
    
    return fig, html_filename

def create_results_summary_table(all_results):
    """Convert list of result dicts to a formatted DataFrame."""
    df = pd.DataFrame(all_results)
    return df

def print_results_table(df, title="Results Summary"):
    """Pretty print results table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")

# ============================================================
# DESCRIPTIVE STATISTICS (for EDA)
# ============================================================
def compute_descriptive_stats(df, stock_name):
    """Compute comprehensive descriptive statistics for a stock."""
    close = df['Close']
    
    stat_dict = {
        'Stock': stock_name,
        'Count': len(close),
        'Mean': round(close.mean(), 4),
        'Median': round(close.median(), 4),
        'Std Dev': round(close.std(), 4),
        'Variance': round(close.var(), 4),
        'Min': round(close.min(), 4),
        'Max': round(close.max(), 4),
        'Range': round(close.max() - close.min(), 4),
        'Skewness': round(close.skew(), 4),
        'Kurtosis': round(close.kurtosis(), 4),
        'Q1 (25%)': round(close.quantile(0.25), 4),
        'Q3 (75%)': round(close.quantile(0.75), 4),
        'IQR': round(close.quantile(0.75) - close.quantile(0.25), 4),
        'CV (%)': round((close.std() / close.mean()) * 100, 4),
    }
    return stat_dict

def perform_normality_tests(data, name="Data"):
    """
    Perform normality tests.
    Returns dict of test results.
    """
    results = {'Name': name}
    
    # Shapiro-Wilk (use subsample if > 5000)
    sample = data[:5000] if len(data) > 5000 else data
    stat_sw, p_sw = stats.shapiro(sample)
    results['Shapiro-Wilk Stat'] = round(stat_sw, 6)
    results['Shapiro-Wilk p-value'] = f'{p_sw:.2e}'
    results['Shapiro-Wilk Normal'] = 'Yes' if p_sw > 0.05 else 'No'
    
    # Kolmogorov-Smirnov
    stat_ks, p_ks = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    results['KS Stat'] = round(stat_ks, 6)
    results['KS p-value'] = f'{p_ks:.2e}'
    results['KS Normal'] = 'Yes' if p_ks > 0.05 else 'No'
    
    # D'Agostino-Pearson (requires n >= 20)
    if len(data) >= 20:
        stat_da, p_da = stats.normaltest(data)
        results['DAgostino Stat'] = round(stat_da, 6)
        results['DAgostino p-value'] = f'{p_da:.2e}'
        results['DAgostino Normal'] = 'Yes' if p_da > 0.05 else 'No'
    
    return results

def perform_stationarity_test(data, name="Data"):
    """
    Augmented Dickey-Fuller test for stationarity.
    """
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data, autolag='AIC')
    
    return {
        'Name': name,
        'ADF Statistic': round(result[0], 6),
        'p-value': f'{result[1]:.2e}',
        'Stationary (p<0.05)': 'Yes' if result[1] < 0.05 else 'No',
        'Critical 1%': round(result[4]['1%'], 4),
        'Critical 5%': round(result[4]['5%'], 4),
        'Critical 10%': round(result[4]['10%'], 4),
    }

# ============================================================
# INTERACTIVE RESULTS VISUALIZATIONS (Plotly)
# ============================================================

def create_interactive_results_dashboard_exp1(results_df, experiment_label, save_dir='figures'):
    """
    Create interactive results dashboard for Experiment 1 (same-stock).
    
    Expected columns: Stock, Model, MSE, RMSE, MAE, MAPE (%), R2
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('RMSE by Stock', 'MAE by Stock', 'R² Score by Stock',
                        'RMSE by Model', 'MAE by Model', 'R² Score by Model'),
        specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Box plots by stock (RMSE, MAE, R²)
    for stock in STOCKS:
        stock_data = results_df[results_df['Stock'] == stock]
        fig.add_trace(
            go.Box(y=stock_data['RMSE'], name=stock, marker_color=STOCK_COLORS.get(stock, '#999')),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=stock_data['MAE'], name=stock, marker_color=STOCK_COLORS.get(stock, '#999'), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=stock_data['R2'], name=stock, marker_color=STOCK_COLORS.get(stock, '#999'), showlegend=False),
            row=1, col=3
        )
    
    # Bar plots by model
    rmse_by_model = results_df.groupby('Model')['RMSE'].mean()
    mae_by_model = results_df.groupby('Model')['MAE'].mean()
    r2_by_model = results_df.groupby('Model')['R2'].mean()
    
    fig.add_trace(
        go.Bar(x=rmse_by_model.index, y=rmse_by_model.values, name='RMSE', 
               marker_color='#0072B2', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=mae_by_model.index, y=mae_by_model.values, name='MAE',
               marker_color='#D55E00', showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=r2_by_model.index, y=r2_by_model.values, name='R²',
               marker_color='#009E73', showlegend=False),
        row=2, col=3
    )
    
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="R² Score", row=1, col=3)
    fig.update_yaxes(title_text="Avg RMSE", row=2, col=1)
    fig.update_yaxes(title_text="Avg MAE", row=2, col=2)
    fig.update_yaxes(title_text="Avg R²", row=2, col=3)
    
    fig.update_layout(
        title=f"<b>{experiment_label} - Results Dashboard</b>",
        height=800,
        showlegend=True,
        template='plotly_white',
        font=dict(size=10)
    )
    
    html_file = f'{save_dir}/{experiment_label}_results_dashboard.html'
    fig.write_html(html_file)
    return fig, html_file

def create_interactive_metrics_heatmap_exp1(results_df, metric, experiment_label, save_dir='figures'):
    """
    Create interactive heatmap for Experiment 1 (Stock vs Model).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pivot_data = results_df.pivot_table(values=metric, index='Stock', columns='Model', aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r' if metric in ['RMSE', 'MAE', 'MSE', 'MAPE (%)'] else 'RdYlGn',
        text=np.round(pivot_data.values, 4),
        texttemplate='%{text:.4f}',
        textfont={"size": 12},
        hovertemplate='Stock: %{y}<br>Model: %{x}<br>' + metric + ': %{z:.4f}<extra></extra>',
        colorbar=dict(title=metric)
    ))
    
    fig.update_layout(
        title=f"<b>{experiment_label} - {metric} by Stock and Model</b>",
        xaxis_title="Model",
        yaxis_title="Stock",
        height=500,
        template='plotly_white',
        font=dict(size=12)
    )
    
    html_file = f'{save_dir}/{experiment_label}_{metric}_heatmap_interactive.html'
    fig.write_html(html_file)
    return fig, html_file

def create_interactive_results_dashboard_exp2(results_df, experiment_label, save_dir='figures'):
    """
    Create interactive results dashboard for Experiment 2 (cross-stock).
    
    Expected columns: Train_Stock, Test_Stock, Model, MSE, RMSE, MAE, MAPE (%), R2
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create scatter plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE vs MAE', 'R² vs RMSE', 'Model Performance (RMSE)', 'Model Performance (R²)'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Scatter plot: RMSE vs MAE (colored by model)
    for model in MODEL_TYPES:
        model_data = results_df[results_df['Model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data['MAE'], y=model_data['RMSE'], 
                       mode='markers', name=model, 
                       marker=dict(size=10, color=MODEL_COLORS.get(model, '#999')),
                       hovertemplate=f'<b>{model}</b><br>MAE: %{{x:.4f}}<br>RMSE: %{{y:.4f}}<extra></extra>'),
            row=1, col=1
        )
    
    # Scatter plot: R² vs RMSE
    for model in MODEL_TYPES:
        model_data = results_df[results_df['Model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data['RMSE'], y=model_data['R2'],
                       mode='markers', name=model, showlegend=False,
                       marker=dict(size=10, color=MODEL_COLORS.get(model, '#999')),
                       hovertemplate=f'<b>{model}</b><br>RMSE: %{{x:.4f}}<br>R²: %{{y:.6f}}<extra></extra>'),
            row=1, col=2
        )
    
    # Bar plots: Average metrics by model
    avg_rmse = results_df.groupby('Model')['RMSE'].mean()
    avg_r2 = results_df.groupby('Model')['R2'].mean()
    
    fig.add_trace(
        go.Bar(x=avg_rmse.index, y=avg_rmse.values, name='RMSE', 
               marker_color=[MODEL_COLORS.get(m, '#999') for m in avg_rmse.index],
               showlegend=False, hovertemplate='Model: %{x}<br>Avg RMSE: %{y:.4f}<extra></extra>'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=avg_r2.index, y=avg_r2.values, name='R²',
               marker_color=[MODEL_COLORS.get(m, '#999') for m in avg_r2.index],
               showlegend=False, hovertemplate='Model: %{x}<br>Avg R²: %{y:.6f}<extra></extra>'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_xaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="R² Score", row=1, col=2)
    fig.update_yaxes(title_text="Average RMSE", row=2, col=1)
    fig.update_yaxes(title_text="Average R²", row=2, col=2)
    
    fig.update_layout(
        title=f"<b>{experiment_label} - Cross-Stock Results Dashboard</b>",
        height=800,
        template='plotly_white',
        font=dict(size=11),
        showlegend=True
    )
    
    html_file = f'{save_dir}/{experiment_label}_crossstock_dashboard.html'
    fig.write_html(html_file)
    return fig, html_file

def create_interactive_transfer_matrix(results_df, metric, experiment_label, save_dir='figures'):
    """
    Create interactive transfer learning matrix heatmap for Experiment 2.
    Rows: Train Stock, Columns: Test Stock
    """
    os.makedirs(save_dir, exist_ok=True)
    
    last_fig = None
    last_html = None
    
    # For each model, create a separate heatmap
    for model in MODEL_TYPES:
        model_data = results_df[results_df['Model'] == model]
        pivot = model_data.pivot_table(values=metric, index='Train_Stock', 
                                       columns='Test_Stock', aggfunc='mean')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn_r' if metric in ['RMSE', 'MAE', 'MSE'] else 'RdYlGn',
            text=np.round(pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 12},
            hovertemplate='Train: %{y}<br>Test: %{x}<br>' + metric + ': %{z:.4f}<extra></extra>',
            colorbar=dict(title=metric)
        ))
        
        fig.update_layout(
            title=f"<b>{experiment_label} - {model} Transfer Learning Matrix ({metric})</b>",
            xaxis_title="Test Stock",
            yaxis_title="Train Stock",
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )
        
        html_file = f'{save_dir}/{experiment_label}_{model}_transfer_matrix_{metric}.html'
        fig.write_html(html_file)
        
        # Keep track of the last figure created (for display)
        last_fig = fig
        last_html = html_file
    
    # Return the last figure created (typically BiLSTM for example display)
    return last_fig, last_html

def create_interactive_metrics_comparison(results_df, experiment_label, save_dir='figures', metrics=['RMSE', 'MAE', 'R2']):
    """
    Create interactive comparison of multiple metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        if metric not in results_df.columns:
            continue
        
        by_model = results_df.groupby('Model')[metric].mean().sort_values()
        
        fig.add_trace(go.Bar(
            x=by_model.index, y=by_model.values,
            name=metric,
            marker_color=['#0072B2', '#D55E00', '#009E73', '#CC79A7'][i % 4],
            text=np.round(by_model.values, 4),
            textposition='outside',
            hovertemplate='Model: %{x}<br>' + metric + ': %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"<b>{experiment_label} - Metrics Comparison by Model</b>",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode='group',
        height=600,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(x=0.99, y=0.99, xanchor='right', yanchor='top')
    )
    
    html_file = f'{save_dir}/{experiment_label}_metrics_comparison.html'
    fig.write_html(html_file)
    return fig, html_file

def create_interactive_model_radar_chart(results_df, experiment_label, save_dir='figures'):
    """
    Create radar chart comparing model performance across metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = go.Figure()
    
    # Normalize metrics for radar chart (0-100 scale, where higher is better)
    for model in MODEL_TYPES:
        model_data = results_df[results_df['Model'] == model]
        
        # Calculate normalized metrics (invert RMSE and MAE so higher is better)
        rmse_norm = max(0, 100 - (model_data['RMSE'].mean() / results_df['RMSE'].max() * 100))
        mae_norm = max(0, 100 - (model_data['MAE'].mean() / results_df['MAE'].max() * 100))
        r2_norm = model_data['R2'].mean() * 100
        mape_norm = max(0, 100 - (model_data.get('MAPE (%)', pd.Series([0])).mean() or 0))
        
        fig.add_trace(go.Scatterpolar(
            r=[rmse_norm, mae_norm, r2_norm, mape_norm],
            theta=['RMSE\n(Lower Better)', 'MAE\n(Lower Better)', 'R²\n(Higher Better)', 'MAPE\n(Lower Better)'],
            fill='toself',
            name=model,
            line_color=MODEL_COLORS.get(model, '#999')
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"<b>{experiment_label} - Model Performance Radar</b>",
        height=700,
        font=dict(size=11),
        template='plotly_white'
    )
    
    html_file = f'{save_dir}/{experiment_label}_model_radar.html'
    fig.write_html(html_file)
    return fig, html_file

print("stock_prediction_utils.py loaded successfully!")
print(f"  ProportionScaler max value: {PROPORTION_SCALER_MAX}")
print(f"  Lookback: {LOOKBACK}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"  Architecture: 2 layers, {UNITS} units, dropout={DROPOUT_RATE}")
print(f"  Stocks: {STOCKS}")
print(f"  Models: {MODEL_TYPES}")

# Setup GPU on module load
setup_gpu()
check_gpu()
