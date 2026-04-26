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
ACTUAL_COLOR = "#EAEAEA"  # Black for actual values

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
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.framealpha': 0.9,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 600,
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

print("stock_prediction_utils.py loaded successfully!")
print(f"  ProportionScaler max value: {PROPORTION_SCALER_MAX}")
print(f"  Lookback: {LOOKBACK}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"  Architecture: 2 layers, {UNITS} units, dropout={DROPOUT_RATE}")
print(f"  Stocks: {STOCKS}")
print(f"  Models: {MODEL_TYPES}")

# Setup GPU on module load
setup_gpu()
check_gpu()
