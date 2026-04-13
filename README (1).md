# A Comparative Analysis of BiLSTM and BiGRU for Stock Price Prediction
## Research Pipeline — Complete Jupyter Notebook Suite

---

## Project Structure

```
├── stock_prediction_utils.py          # Shared utility module (REQUIRED)
├── Notebook_0_EDA.ipynb               # Exploratory Data Analysis & Statistics
├── Notebook_1_Exp1_80_20.ipynb        # Exp1: Same-stock prediction (80/20)
├── Notebook_2_Exp2_80_20.ipynb        # Exp2: Cross-stock prediction (80/20)
├── Notebook_3_Exp3_80_20.ipynb        # Exp3: Different timeframe training (80/20)
├── Notebook_4_Exp4_80_20.ipynb        # Exp4: Multi-stock training (80/20)
├── Notebook_5_Exp1_70_30.ipynb        # Exp1: Same-stock prediction (70/30)
├── Notebook_6_Exp2_70_30.ipynb        # Exp2: Cross-stock prediction (70/30)
├── Notebook_7_Exp3_70_30.ipynb        # Exp3: Different timeframe training (70/30)
├── Notebook_8_Exp4_70_30.ipynb        # Exp4: Multi-stock training (70/30)
├── Notebook_9_Summary.ipynb           # Final comparison & summary
├── TLKM_Daily_Data.xlsx               # Data files (place here)
├── BBCA_Daily_Data.xlsx
├── ASII_Daily_Data.xlsx
├── UNVR_Daily_Data.xlsx
├── tlkm_weekly_data.xlsx
├── bbca_weekly_data.xlsx
├── asii_weekly_data.xlsx
├── unvr_weekly_data.xlsx
├── tlkm_monthly_data.xlsx
├── bbca_monthly_data.xlsx
├── asii_monthly_data.xlsx
├── unvr_monthly_data.xlsx
├── tlkm_yearly_data.xlsx
├── bbca_yearly_data.xlsx
├── asii_yearly_data.xlsx
├── unvr_yearly_data.xlsx
├── figures/                           # Generated figures (600 DPI PNG)
├── models/                            # Saved Keras models (.keras)
└── results/                           # Result CSV files
```

## Setup

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow openpyxl scipy statsmodels
```

### 2. Place Data Files
Place all 16 `.xlsx` data files in the **same directory** as the notebooks and `stock_prediction_utils.py`.

### 3. Run Order
**Run notebooks in order (0 → 9).** Each notebook is independent except:
- All notebooks import from `stock_prediction_utils.py`
- Notebook 9 reads CSV results from Notebooks 1–8

## Configuration

All hyperparameters are defined in `stock_prediction_utils.py`:

| Parameter | Value |
|-----------|-------|
| Lookback Window | 60 timesteps |
| Epochs | 200 (with early stopping, patience=20) |
| Batch Size | 64 |
| Hidden Units | 64 per layer |
| Layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam (lr=0.001) |
| Scaler | ProportionScaler (÷ 10,501) |
| Seed | 42 |

## Experiments

### Experiment 1: Same-Stock Prediction
Train and test on the same stock's daily data. Tests each model's fitting ability.

### Experiment 2: Cross-Stock Prediction
Train on Stock A, predict on Stock B. Tests generalization across different stocks.

### Experiment 3: Different Timeframe Training
Train on weekly/monthly/yearly data, predict on daily data. Tests temporal transfer.

### Experiment 4: Multi-Stock Training
Train on 3 stocks combined, predict on the 4th. Tests multi-source learning.

Each experiment runs with **both 80/20 and 70/30** train/test splits.

## Data Leakage Prevention
- Chronological train/test split (no shuffling of the split)
- ProportionScaler uses a fixed constant (10,501) — no information leakage
- Validation set taken from the end of training data (temporal order preserved)
- Cross-timeframe experiments only use training timeframe data up to the daily split date

## Outputs
- **Figures**: 600 DPI PNG, IEEE-compatible formatting (serif font, clear labels)
- **Models**: Best (ModelCheckpoint on val_loss) and Last saved per experiment
- **Results**: CSV files with MSE, RMSE, MAE, MAPE, R² for every experiment

## Notes
- Training ~200 experiments × 200 epochs will take significant time. Consider using a GPU.
- Early stopping (patience=20) prevents unnecessary epochs.
- Memory is cleared between model trainings to prevent OOM errors.
