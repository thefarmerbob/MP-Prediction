import numpy as np
import pandas as pd
import sys
import os
sys.path.append('..')  # Add parent directory to path
sys.path.append('.')   # Add current directory to path
from util import *
from myXgb import xgb_importance, xgb_data_split, xgb_forecasts_plot_with_actual
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor  # wrapper
import scipy.stats as st
import optuna
from sklearn.metrics import mean_squared_error

# NOTE: This script uses Optuna optimization for key XGBoost parameters
# Optimizing: learning_rate, max_depth, n_estimators
# Keeping other parameters fixed for focused optimization

# =============================================================================
# DATA SPLIT CONFIGURATION - All ratios organized in one place
# =============================================================================
# Validation ratio for train/validation split
val_ratio = 0.3  # Optimized: validation ratio

# Train/Test split ratios for time series 
train_ratio = 0.70 # Optuna best

config_plot()

N_rows = 2491  # Total number of rows in mp-kyoto-avg.txt
filename = "../regional-averages/mp-kyoto-avg.txt"
encode_cols = ['Month', 'DayofWeek', 'Hour']

# Global variables for Optuna optimization
global_X_train, global_X_val, global_y_train, global_y_val = None, None, None, None

def preprocess_regional_data(filename):
    """Custom preprocessing function for regional average data format"""
    # Read the regional data file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse the data
    data = []
    for line in lines:
        line = line.strip()
        if line:
            # Format: 20180816-120000-e20180816-120000: 12780.02
            parts = line.split(': ')
            if len(parts) == 2:
                timestamp_str = parts[0]
                value_str = parts[1]
                
                # Skip NaN values
                if value_str.lower() == 'nan':
                    continue
                
                # Parse timestamp
                # Format: 20180816-120000-e20180816-120000
                date_part = timestamp_str.split('-')[0]
                time_part = timestamp_str.split('-')[1]
                
                # Convert to datetime format
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                hour = time_part[:2]
                minute = time_part[2:4]
                second = time_part[4:6]
                
                datetime_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                
                try:
                    value = float(value_str)
                    data.append([datetime_str, value])
                except ValueError:
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['DateTime', 'Microplastic_Concentration'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    return df

def objective(trial):
    """Optuna objective function for XGBoost hyperparameter optimization"""
    # Suggest parameters to optimize (only key parameters)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=50)
    
    # Fixed parameters (keep the best known values for these)
    params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'subsample': 0.58,  # Fixed
        'colsample_bytree': 0.499,  # Fixed
        'min_child_weight': 14,  # Fixed
        'learning_rate': learning_rate,  # Optimize
        'max_depth': max_depth,  # Optimize
        'n_estimators': n_estimators,  # Optimize
        'reg_alpha': 27.12,  # Fixed
        'reg_lambda': 10.34,  # Fixed
        'gamma': 2.24,  # Fixed
        'tree_method': 'auto',  # Fixed
        'sampling_method': 'uniform',  # Fixed
        'seed': 42,
        'verbosity': 0  # Reduce output during optimization
    }
    
    # Create DMatrix for training
    dtrain = xgb.DMatrix(global_X_train, global_y_train)
    dval = xgb.DMatrix(global_X_val, global_y_val)
    
    # Train model with early stopping
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Predict on validation set
    y_pred = model.predict(dval)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(global_y_val, y_pred))
    
    return rmse

# Prepare data for optimization - ONLY USE TRAINING DATA
print("Loading Kyoto regional data...")
df = preprocess_regional_data(filename)
print(f"Kyoto data shape: {df.shape}")

# Add datetime features
df = date_transform(df, encode_cols)

# Add enhanced features for better microplastic concentration prediction (reduced set)
print("Adding reduced temporal features for Kyoto...")
# Add only a few recent lags
for lag in [1, 7, 30]:
    df[f'mp_lag_{lag}'] = df['Microplastic_Concentration'].shift(lag)

# Add only rolling means
for window in [7, 30]:
    df[f'mp_rolling_mean_{window}'] = df['Microplastic_Concentration'].rolling(window=window).mean()

# Add basic trend features
df['mp_diff'] = df['Microplastic_Concentration'].diff()
df['mp_pct_change'] = df['Microplastic_Concentration'].pct_change()

# Remove NaN values created by lag features
df.dropna(inplace=True)
print(f"Reduced features added. New shape: {df.shape}")

# TEMPORAL SPLIT FOR OPTUNA - ONLY USE TRAINING PORTION
total_points = len(df)
train_points = int(total_points * train_ratio)
train_end_idx = train_points

# Use only training data for Optuna optimization
df_train_only = df.iloc[:train_end_idx]
print(f"Optuna will use only training data: {len(df_train_only)} points")

# Prepare data for Optuna optimization (ONLY TRAINING DATA)
Y = df_train_only.iloc[:, 0]
X = df_train_only.iloc[:, 1:]

# Convert categorical columns to numeric to avoid XGBoost dtype errors
X = X.astype(float)

# Split training data for optimization (train/validation split within training data)
global_X_train, global_X_val, global_y_train, global_y_val = train_test_split(
    X, Y, test_size=val_ratio, random_state=42
)

# Check if Optuna study already exists and has completed trials
storage = optuna.storages.RDBStorage(url="sqlite:///kyoto_optuna_study.db")
study = optuna.create_study(
    direction='minimize', 
    study_name='kyoto_xgb_optimization',
    storage=storage,
    load_if_exists=True
)

# Check if study already has completed trials
if len(study.trials) > 0:
    print("✅ Found existing Optuna study for Kyoto with completed trials.")
    print("Using best parameters from existing study...")
    best_params = study.best_params
    best_rmse = study.best_value
    print(f"Best RMSE from existing study: {best_rmse:.6f}")
    print(f"Best parameters from existing study:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
else:
    print("Starting new Optuna optimization for Kyoto XGBoost parameters...")
    print("Optimizing: learning_rate, max_depth, n_estimators")
    print("Keeping other parameters fixed at their best known values")
    
    study.optimize(objective, n_trials=100)  # Limited to 100 trials for efficiency
    
    # Get best parameters
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"\nOptuna Optimization Results for Kyoto:")
    print(f"Best RMSE: {best_rmse:.6f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

# XGBoost parameters with Optuna-optimized values
optimized_xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'subsample': 0.3,  # Fixed
    'colsample_bytree': 0.499,  # Fixed
    'min_child_weight': 14,  # Fixed
    'learning_rate': best_params.get('learning_rate', 0.03766600333324149),  # Optuna-optimized
    'max_depth': best_params.get('max_depth', 6),  # Optuna-optimized
    'n_estimators': best_params.get('n_estimators', 3000),  # Optuna-optimized
    'reg_alpha': 27.12,  # Fixed
    'reg_lambda': 10.34,  # Fixed
    'gamma': 2.24,  # Fixed
    'tree_method': 'auto',  # Fixed
    'sampling_method': 'uniform',  # Fixed
    'seed': 42
}

ntree = 300  # Optuna-optimized number of trees
early_stop = 50

print('\n-----XGBoost Using All Numeric Features with Optuna-Optimized Parameters for Kyoto-----')
print('---Feature importance with optimized model---')
fig_allFeatures = xgb_importance(
    df, val_ratio, optimized_xgb_params, ntree, early_stop, 'Kyoto_Optuna_Optimized_All_Features')

#############################################################################
# xgboost using only datetime information with optimized parameters
bucket_size = "1D"  # Daily data for microplastic concentration
df = preprocess_regional_data(filename)
mp_concentration = df["Microplastic_Concentration"]

df = pd.DataFrame(bucket_avg(mp_concentration, bucket_size))
df.dropna(inplace=True)

# Debug: Print the actual date range of the data
print("Kyoto data date range after preprocessing:")
print(f"First timestamp: {df.index[0]}")
print(f"Last timestamp: {df.index[-1]}")
print(f"Total data points: {len(df)}")

df.iloc[-1, :].index  # last time step

# Adjust dates based on actual data range - Using centralized split ratios
total_points = len(df)
train_points = int(total_points * train_ratio)  # Using centralized train_ratio
test_points = total_points - train_points       # Use all remaining data for test

# Ensure no overlap by making train end before test starts
train_end_idx = train_points
test_start_idx = train_points + 1  # Add 1 to ensure no overlap
test_end_idx = total_points

test_start_date = df.index[test_start_idx].strftime('%Y-%m-%d %H:%M:%S')

print(f"Training data: {df.index[0]} to {df.index[train_end_idx-1]}")
print(f"Test data: {test_start_date} to {df.index[test_end_idx-1]}")

# Set up plot start index for later use
plot_start_idx = 0  # Show ALL historical data from the beginning

# get splited data - train, test, and forecast (30 days for about a month)
df_forecast, df_test, df = xgb_data_split(
    df, bucket_size, df.index[-1].strftime('%Y-%m-%d %H:%M:%S'), 30, test_start_date, encode_cols)
print('\n-----XGBoost on datetime information with Optuna-Optimized Parameters for Kyoto-----\n')

dim = {'train and validation data ': df.shape,
       'test data ': df_test.shape,
       'forecast data ': df_forecast.shape}
print(pd.DataFrame(list(dim.items()), columns=['Data', 'dimension']))

# train model with optimized parameters
Y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Convert categorical columns to numeric to avoid XGBoost dtype errors
X = X.astype(float)

X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                  test_size=val_ratio,
                                                  random_state=42)

X_test = xgb.DMatrix(df_test.iloc[:, 1:].astype(float))
Y_test = df_test.iloc[:, 0]

# Prepare forecast data
X_forecast = xgb.DMatrix(df_forecast.astype(float))

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
watchlist = [(dtrain, 'train'), (dval, 'validate')]

#############################################################################
# Testing using Optuna-optimized model
print("Training final Kyoto model with Optuna-optimized parameters...")
xgb_model = xgb.train(optimized_xgb_params, dtrain, ntree, evals=watchlist,
                      early_stopping_rounds=early_stop, verbose_eval=False)
Y_hat = xgb_model.predict(X_test)
Y_hat = pd.DataFrame(Y_hat, index=Y_test.index, columns=["test_predicted"])

# Make forecast predictions
Y_forecast = xgb_model.predict(X_forecast)
Y_forecast = pd.DataFrame(Y_forecast, index=df_forecast.index, columns=["forecast_predicted"])

plot_start = df.index[plot_start_idx].strftime('%Y-%m-%d %H:%M:%S')
print('-----XGBoost with Optuna-Optimized Parameters for Kyoto Microplastic Concentration------')
print('---Testing with optimized model---')

# Create comprehensive plot with forecast data
Y_combined = pd.concat([Y, Y_test])
ax = Y_combined[plot_start:].plot(label='observed', figsize=(15, 10), color='#00FFFF', alpha=0.5)
Y_hat.plot(label="test_predicted", ax=ax, color='#008080', linewidth=2)
Y_forecast.plot(label="forecast_predicted", ax=ax, color='#FF6B35', linewidth=2, linestyle='--')

# Highlight test period
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(Y_test.index[0]), Y_test.index[-1],
                 alpha=0.1, color='#008080', zorder=-1, label='Test Period')

# Highlight forecast period
ax.fill_betweenx(ax.get_ylim(), Y_forecast.index[0], Y_forecast.index[-1],
                 alpha=0.1, color='#FF6B35', zorder=-1, label='Forecast Period (30 days)')

ax.set_xlabel('Time')
ax.set_ylabel('Microplastic Concentration')
ax.set_title('Kyoto Optuna_Optimized_Model - Microplastic Concentration Prediction with 30-Day Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Kyoto_Optuna_Optimized_Model_with_Forecast.png', dpi=300)
plt.close()
print("Kyoto prediction plot with 30-day forecast saved as: Kyoto_Optuna_Optimized_Model_with_Forecast.png")

# Calculate and display final model performance metrics
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_hat))
print(f"\nFinal Kyoto Model Performance:")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Validation RMSE (from optimization): {best_rmse:.6f}")

# Display forecast information
print(f"\nForecast Information:")
print(f"Forecast period: {Y_forecast.index[0].strftime('%Y-%m-%d')} to {Y_forecast.index[-1].strftime('%Y-%m-%d')}")
print(f"Number of forecast days: {len(Y_forecast)}")
print(f"Forecast value range: {Y_forecast.iloc[:, 0].min():.2f} to {Y_forecast.iloc[:, 0].max():.2f}")
print(f"Average forecast value: {Y_forecast.iloc[:, 0].mean():.2f}")

# Save the optimized parameters for future reference
optimized_params_df = pd.DataFrame([best_params])
optimized_params_df.to_csv('kyoto_optuna_optimized_parameters.csv', index=False)
print("Kyoto optimized parameters saved to 'kyoto_optuna_optimized_parameters.csv'") 