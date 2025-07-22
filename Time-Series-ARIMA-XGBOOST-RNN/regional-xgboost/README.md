# Regional and Global XGBoost Microplastic Prediction Models

This directory contains separate XGBoost prediction models for the global average and each regional microplastic concentration dataset. Each model is optimized specifically for its respective dataset using Optuna hyperparameter optimization.

## Files Overview

### 1. `Gpower_Xgb_Main_global.py`
- **Purpose**: Predicts microplastic concentration for the Global average
- **Data Source**: `../mp-avg/mp-global-avg.txt`
- **Features**: 
  - Temporal features (lags: 1, 7, 30 days)
  - Rolling means (7, 30 days)
  - Trend features (difference, percentage change)
  - Datetime features (Month, DayofWeek, Hour)
- **Outputs**:
  - `Global_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `Global_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `global_optuna_optimized_parameters.csv` - Optimized parameters
  - `global_optuna_study.db` - Optuna study database

### 2. `Gpower_Xgb_Main_kyoto.py`
- **Purpose**: Predicts microplastic concentration for the Kyoto region
- **Data Source**: `../regional-averages/mp-kyoto-avg.txt`
- **Features**: 
  - Temporal features (lags: 1, 7, 30 days)
  - Rolling means (7, 30 days)
  - Trend features (difference, percentage change)
  - Datetime features (Month, DayofWeek, Hour)
- **Outputs**:
  - `Kyoto_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `Kyoto_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `kyoto_optuna_optimized_parameters.csv` - Optimized parameters
  - `kyoto_optuna_study.db` - Optuna study database

### 3. `Gpower_Xgb_Main_osaka.py`
- **Purpose**: Predicts microplastic concentration for the Osaka region
- **Data Source**: `../regional-averages/mp-osaka-avg.txt`
- **Features**: Same as Kyoto model
- **Outputs**:
  - `Osaka_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `Osaka_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `osaka_optuna_optimized_parameters.csv` - Optimized parameters
  - `osaka_optuna_study.db` - Optuna study database

### 4. `Gpower_Xgb_Main_tokyo.py`
- **Purpose**: Predicts microplastic concentration for the Tokyo region
- **Data Source**: `../regional-averages/mp-tokyo-avg.txt`
- **Features**: Same as Kyoto model
- **Outputs**:
  - `Tokyo_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `Tokyo_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `tokyo_optuna_optimized_parameters.csv` - Optimized parameters
  - `tokyo_optuna_study.db` - Optuna study database

### 5. `Gpower_Xgb_Main_tsushima.py`
- **Purpose**: Predicts microplastic concentration for the Tsushima region
- **Data Source**: `../regional-averages/mp-tsushima-avg.txt`
- **Features**: Same as Kyoto model
- **Outputs**:
  - `Tsushima_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `Tsushima_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `tsushima_optuna_optimized_parameters.csv` - Optimized parameters
  - `tsushima_optuna_study.db` - Optuna study database

## Data Format

The regional data files have a different format than the global average:
- **Format**: `YYYYMMDD-HHMMSS-eYYYYMMDD-HHMMSS: value`
- **Example**: `20180816-120000-e20180816-120000: 12780.02`
- **Handling**: NaN values are automatically skipped during preprocessing

## Model Configuration

All models use the same optimized configuration:
- **Train/Test Split**: 70% training, 30% testing
- **Validation Split**: 30% of training data for validation
- **Optimization**: Optuna with 100 trials
- **Parameters Optimized**: learning_rate, max_depth, n_estimators
- **Fixed Parameters**: Based on best known values from global model

## Usage

### Individual Models

To run any of the models individually:

```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost
python Gpower_Xgb_Main_global.py   # For Global
python Gpower_Xgb_Main_kyoto.py    # For Kyoto
python Gpower_Xgb_Main_osaka.py    # For Osaka
python Gpower_Xgb_Main_tokyo.py    # For Tokyo
python Gpower_Xgb_Main_tsushima.py # For Tsushima
```

### Batch Processing

To run all regional models sequentially:

```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost
python run_all_regional_models.py
```

This will execute all five models in order and provide a summary of results.

## Key Features

1. **Custom Preprocessing**: Each model includes a `preprocess_regional_data()` function that handles the specific regional data format
2. **Smart Optuna Optimization**: Each region gets its own optimized hyperparameters with intelligent caching:
   - If an Optuna study already exists with completed trials, it uses the best parameters from the existing study
   - If no study exists, it runs a new optimization with 100 trials
3. **Feature Engineering**: Temporal features, rolling statistics, and trend analysis
4. **Performance Metrics**: RMSE for both validation and test sets
5. **Visualization**: Prediction plots and feature importance analysis

## Dependencies

- numpy
- pandas
- xgboost
- optuna
- matplotlib
- scikit-learn
- scipy

## Notes

- Each model creates its own Optuna study database to avoid conflicts
- The preprocessing function handles the regional data format automatically
- All models use the same feature engineering approach for consistency
- Results are saved with region-specific naming to avoid conflicts 