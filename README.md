# MP-Prediction: Microplastic Prediction using CYGNSS Data

This project implements machine learning models to predict microplastic concentrations using CYGNSS (Cyclone Global Navigation Satellite System) data.

## Project Structure

```
MP-Prediction/
├── Time-Series-ARIMA-XGBOOST-RNN/
│   ├── arima/                    # ARIMA time series models
│   ├── gradient-boost/           # XGBoost models
│   ├── lstm/                     # LSTM neural network models
│   ├── regional-xgboost/         # Regional XGBoost models
│   ├── mp-avg/                   # Global microplastic averages
│   ├── regional-averages/        # Regional microplastic averages
│   └── timeseries_plots/         # Time series visualization
├── CYGNSS-data/                  # CYGNSS satellite data files
└── cygnss_mp_demo.py            # Main demo script
```

## Features

- **Multiple ML Models**: ARIMA, XGBoost, and LSTM implementations
- **Regional Analysis**: Separate models for different geographical regions
- **Time Series Analysis**: Comprehensive time series modeling and visualization
- **Optuna Optimization**: Hyperparameter optimization for all models
- **Data Processing**: CYGNSS data processing and feature engineering

## Models

### 1. ARIMA Models
- Location: `Time-Series-ARIMA-XGBOOST-RNN/arima/`
- Files: `Gpower_Arima_Main.py`, `myArima.py`
- Purpose: Time series forecasting using ARIMA methodology

### 2. XGBoost Models
- Location: `Time-Series-ARIMA-XGBOOST-RNN/gradient-boost/`
- Files: `Gpower_Xgb_Main.py`, `myXgb.py`
- Features: Optuna hyperparameter optimization
- Output: Optimized models and performance plots

### 3. LSTM Models
- Location: `Time-Series-ARIMA-XGBOOST-RNN/lstm/`
- Files: `lstm_Main.py`, `lstm_optuna.py`, `lstm_optuna_quick.py`
- Purpose: Deep learning time series prediction

### 4. Regional Models
- Location: `Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost/`
- Purpose: Region-specific microplastic prediction models
- Features: Combined regional analysis and visualization

## Data

The project uses CYGNSS satellite data stored in the `CYGNSS-data/` directory:
- NetCDF files containing microplastic concentration data
- Time series data from 2018-2019
- Global and regional coverage

## Usage

### Main Demo
```bash
python cygnss_mp_demo.py
```

### Individual Models

#### ARIMA
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/arima/
python Gpower_Arima_Main.py
```

#### XGBoost
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/gradient-boost/
python Gpower_Xgb_Main.py
```

#### LSTM
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/lstm/
python lstm_Main.py
```

## Dependencies

The project uses a virtual environment with the following key dependencies:
- numpy
- pandas
- scikit-learn
- xgboost
- tensorflow/keras
- matplotlib
- seaborn
- optuna
- netCDF4

## Results

The project generates:
- Optimized model files (`.json`, `.pkl`)
- Performance plots and visualizations
- Time series analysis plots
- Regional comparison charts
- Clustering analysis results

## License

This project is for research purposes in microplastic prediction using satellite data.

## Contact

For questions or contributions, please contact the project maintainers. 