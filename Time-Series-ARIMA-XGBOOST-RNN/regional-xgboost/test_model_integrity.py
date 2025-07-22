import numpy as np
import pandas as pd
import sys
import os
sys.path.append('..')  # Add parent directory to path
sys.path.append('.')   # Add current directory to path
from util import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

print("=== TESTING MODEL INTEGRITY ===")

# Configuration
train_ratio = 0.70
filename = "../regional-averages/mp-tokyo-avg.txt"
encode_cols = ['Month', 'DayofWeek', 'Hour']
bucket_size = "1D"

# Load and preprocess data
df = preprocess_regional_data(filename)
mp_concentration = df["Microplastic_Concentration"]
df = pd.DataFrame(bucket_avg(mp_concentration, bucket_size))
df.dropna(inplace=True)

print(f"Total data points: {len(df)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Calculate split points
total_points = len(df)
train_points = int(total_points * train_ratio)
test_points = total_points - train_points

train_end_idx = train_points
test_start_idx = train_points + 1  # Ensure no overlap
test_end_idx = total_points

# Simple temporal split
df_train = df.iloc[:train_end_idx]
df_test = df.iloc[test_start_idx:test_end_idx]

print(f"\nSplit:")
print(f"Train: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} points)")
print(f"Test: {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} points)")

# Apply date transformations
df_train = date_transform(df_train, encode_cols)
df_test = date_transform(df_test, encode_cols)

# Prepare data
Y_train = df_train.iloc[:, 0]
X_train = df_train.iloc[:, 1:].astype(float)
Y_test = df_test.iloc[:, 0]
X_test = df_test.iloc[:, 1:].astype(float)

print(f"\nFeature shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# Check for any obvious data leakage in features
print(f"\nFeature analysis:")
print(f"X_train columns: {list(X_train.columns)}")

# Look for any features that might contain future information
suspicious_features = []
for col in X_train.columns:
    if 'lag' in col.lower() or 'future' in col.lower() or 'next' in col.lower():
        suspicious_features.append(col)

if suspicious_features:
    print(f"⚠️  Suspicious features found: {suspicious_features}")
else:
    print("✅ No obviously suspicious features found")

# Train a simple model to test performance
print(f"\nTraining simple XGBoost model...")

# Simple parameters
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'seed': 42,
    'verbosity': 0
}

# Split train into train/validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, Y_train, test_size=0.3, random_state=42)

# Train model
dtrain = xgb.DMatrix(X_train_split, y_train_split)
dval = xgb.DMatrix(X_val, y_val)
dtest = xgb.DMatrix(X_test, Y_test)

watchlist = [(dtrain, 'train'), (dval, 'validate')]

model = xgb.train(params, dtrain, 100, evals=watchlist, 
                  early_stopping_rounds=10, verbose_eval=False)

# Make predictions
train_pred = model.predict(dtrain)
val_pred = model.predict(dval)
test_pred = model.predict(dtest)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_split, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_pred))

print(f"\nModel Performance:")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Check if test performance is suspiciously good
if test_rmse < val_rmse * 0.8:  # Test RMSE is 20% better than validation
    print("⚠️  WARNING: Test performance is suspiciously good!")
    print("This might indicate data leakage or overfitting.")
elif test_rmse > val_rmse * 1.5:  # Test RMSE is 50% worse than validation
    print("⚠️  WARNING: Test performance is much worse than validation!")
    print("This might indicate overfitting to validation set.")
else:
    print("✅ Test performance looks reasonable compared to validation.")

# Check data ranges
print(f"\nData ranges:")
print(f"Train target range: {Y_train.min():.2f} to {Y_train.max():.2f}")
print(f"Test target range: {Y_test.min():.2f} to {Y_test.max():.2f}")

# Check if test data is in a completely different range
train_mean = Y_train.mean()
test_mean = Y_test.mean()
train_std = Y_train.std()
test_std = Y_test.std()

print(f"\nStatistical comparison:")
print(f"Train mean: {train_mean:.2f}, std: {train_std:.2f}")
print(f"Test mean: {test_mean:.2f}, std: {test_std:.2f}")

if abs(test_mean - train_mean) > 2 * train_std:
    print("⚠️  WARNING: Test data has very different distribution!")
    print("This might explain poor generalization.")
else:
    print("✅ Test data distribution looks similar to training data.") 