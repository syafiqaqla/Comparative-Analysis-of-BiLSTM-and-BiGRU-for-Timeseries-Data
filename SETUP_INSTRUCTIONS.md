# Setup Instructions for Stock Prediction Pipeline

This document provides instructions for setting up the stock prediction pipeline on a new machine.

## Quick Start

### Option 1: Using pip (Recommended for beginners)

1. **Clone/Download the repository**
   ```bash
   git clone <your-repo-url>
   cd "Ultimate Code"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

### Option 2: Using Conda (Recommended for data science)

1. **Clone/Download the repository**
   ```bash
   git clone <your-repo-url>
   cd "Ultimate Code"
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate environment**
   ```bash
   conda activate stock-prediction
   ```

4. **Run Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

## Project Structure

```
Ultimate Code/
├── Notebook_0_EDA.ipynb                    # Exploratory Data Analysis
├── Notebook_1_Exp1_80_20.ipynb            # Experiment 1: Same-stock (80/20 split)
├── Notebook_2_Exp2_80_20.ipynb            # Experiment 2: Cross-stock zero-shot (80/20)
├── Notebook_3_Exp3_80_20.ipynb            # Experiment 3: Different timeframes (80/20)
├── Notebook_4_Exp4_80_20.ipynb            # Experiment 4: [Custom/Alternative] (80/20)
├── Notebook_5_Exp1_70_30.ipynb            # Experiment 1: Same-stock (70/30 split)
├── Notebook_6_Exp2_70_30.ipynb            # Experiment 2: Cross-stock zero-shot (70/30)
├── Notebook_7_Exp3_70_30.ipynb            # Experiment 3: Different timeframes (70/30)
├── Notebook_8_Exp4_70_30.ipynb            # Experiment 4: [Custom/Alternative] (70/30)
├── Notebook_9_Summary.ipynb                # Summary and results analysis
├── stock_prediction_utils.py               # Shared utility module
├── requirements.txt                        # pip dependencies
├── environment.yml                         # conda environment file
├── dataset/                                # Input data (Excel files)
├── models/                                 # Trained model checkpoints
│   ├── Exp1_80_20/
│   ├── Exp2_80_20/
│   ├── Exp3_80_20/
│   ├── Exp4_80_20/
│   ├── Exp1_70_30/
│   ├── Exp2_70_30/
│   ├── Exp3_70_30/
│   └── Exp4_70_30/
├── figures/                                # Generated visualizations
│   ├── eda/
│   ├── Exp1_80_20/
│   ├── Exp2_80_20/
│   ├── Exp3_80_20/
│   ├── Exp4_80_20/
│   ├── Exp1_70_30/
│   ├── Exp2_70_30/
│   ├── Exp3_70_30/
│   └── Exp4_70_30/
├── results/                                # CSV results files
└── README.md
```

## Dependencies Overview

### Core Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| numpy | Numerical computing | ≥1.21.0 |
| pandas | Data manipulation | ≥1.3.0 |
| scipy | Scientific computing | ≥1.7.0 |
| scikit-learn | Machine learning utilities | ≥1.0.0 |
| tensorflow | Deep learning framework | ≥2.10.0 |
| matplotlib | Plotting library | ≥3.4.0 |
| seaborn | Statistical visualization | ≥0.11.0 |
| plotly | Interactive plotting | ≥5.0.0 |
| openpyxl | Excel file support | ≥3.6.0 |
| jupyter | Notebook environment | ≥1.0.0 |

## GPU Support (Optional)

If you have an NVIDIA GPU and want to accelerate training:

### Windows/macOS/Linux with CUDA

1. **Install CUDA Toolkit** (11.8 recommended for TensorFlow 2.10+)
   - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

2. **Install cuDNN** (8.6+ recommended)
   - [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

3. **Verify GPU setup in Python**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### Using Conda (Automatic CUDA setup)

Uncomment the GPU lines in `environment.yml`:
```yaml
dependencies:
  - cudatoolkit=11.8
  - cudnn=8.6
```

Then recreate the environment:
```bash
conda env create -f environment.yml --force
```

## Troubleshooting

### Common Issues

**Issue: TensorFlow not found**
```bash
pip install --upgrade tensorflow
```

**Issue: Excel file not found**
- Ensure `dataset/` folder contains `.xlsx` files with format: `{stock}_daily_data.xlsx`
- Check stock names match: TLKM, BBCA, ASII, UNVR

**Issue: ModuleNotFoundError for openpyxl**
```bash
pip install openpyxl
```

**Issue: Jupyter notebooks won't start**
```bash
pip install --upgrade jupyter jupyterlab
jupyter notebook --NotebookApp.iopub_msg_rate_limit=1e10
```

**Issue: Out of memory (OOM) during training**
- Reduce `BATCH_SIZE` in notebooks (currently 64)
- Use GPU instead of CPU (see GPU Support section)

## Running the Pipeline

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Run notebooks in order**
   - `Notebook_0_EDA.ipynb` - Exploratory analysis
   - `Notebook_1_Exp1_80_20.ipynb` - Train baseline models (80/20)
   - `Notebook_5_Exp1_70_30.ipynb` - Train baseline models (70/30)
   - `Notebook_2_Exp2_80_20.ipynb` - Cross-stock transfer learning (80/20)
   - `Notebook_6_Exp2_70_30.ipynb` - Cross-stock transfer learning (70/30)
   - `Notebook_3_Exp3_80_20.ipynb` - Multi-timeframe training (80/20)
   - `Notebook_7_Exp3_70_30.ipynb` - Multi-timeframe training (70/30)
   - `Notebook_9_Summary.ipynb` - Comparative analysis

## Updating Dependencies

### If pip version becomes outdated

```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

### If conda environment needs update

```bash
conda env update -f environment.yml --prune
```

## Exporting Current Environment

To save your current working environment for sharing:

### pip
```bash
pip freeze > requirements-lock.txt
```

### conda
```bash
conda env export > environment-lock.yml
```

## Additional Notes

- **Reproducibility**: All notebooks set random seeds for reproducibility
- **Data Format**: Stock prices should be in `.xlsx` files with `Date` and `Close` columns
- **Scaler**: Uses ProportionScaler (divides by BBCA ATH = 10,501) - fixed constant, no data leakage
- **Models**: BiLSTM, BiGRU, LSTM, GRU implemented with 64 units, 0.2 dropout
- **Metrics**: MSE, RMSE, MAE, MAPE (%), R² Score

## Support

For issues or questions:
1. Check `stock_prediction_utils.py` for function documentation
2. Review notebook comments for methodology details
3. Check TensorFlow documentation: https://www.tensorflow.org/
