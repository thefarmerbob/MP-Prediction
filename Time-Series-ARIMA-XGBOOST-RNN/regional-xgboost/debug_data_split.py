import numpy as np
import pandas as pd
import sys
import os
sys.path.append('..')  # Add parent directory to path
sys.path.append('.')   # Add current directory to path
from util import *
from myXgb import xgb_data_split

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

# Configuration
train_ratio = 0.70
filename = "../regional-averages/mp-tokyo-avg.txt"
encode_cols = ['Month', 'DayofWeek', 'Hour']
bucket_size = "1D"

print("=== DEBUGGING DATA SPLIT FOR TOKYO ===")

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

# Ensure no overlap by making train end before test starts
train_end_idx = train_points
test_start_idx = train_points + 1  # Add 1 to ensure no overlap
test_end_idx = total_points

test_start_date = df.index[test_start_idx].strftime('%Y-%m-%d %H:%M:%S')
unseen_start_date = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')

print(f"\nSplit Configuration:")
print(f"Train points: {train_points} ({train_ratio*100}%)")
print(f"Test points: {test_points} ({(1-train_ratio)*100}%)")
print(f"Train data: {df.index[0]} to {df.index[train_end_idx-1]}")
print(f"Test data: {test_start_date} to {df.index[test_end_idx-1]}")

# Use the same splitting logic as the main script
_, df_test, df_train = xgb_data_split(
    df, bucket_size, unseen_start_date, 0, test_start_date, encode_cols)

print(f"\nAfter xgb_data_split:")
print(f"Train data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")

# Check for overlap
train_dates = set(df_train.index)
test_dates = set(df_test.index)
overlap = train_dates.intersection(test_dates)

print(f"\nOverlap Analysis:")
print(f"Train dates: {len(train_dates)}")
print(f"Test dates: {len(test_dates)}")
print(f"Overlapping dates: {len(overlap)}")

if len(overlap) > 0:
    print("⚠️  WARNING: DATA LEAKAGE DETECTED!")
    print(f"Overlapping dates: {sorted(overlap)[:10]}...")  # Show first 10
else:
    print("✅ No data leakage detected!")

# Check the actual date ranges
print(f"\nDate Ranges:")
print(f"Train: {df_train.index[0]} to {df_train.index[-1]}")
print(f"Test: {df_test.index[0]} to {df_test.index[-1]}")

# Verify chronological order
if df_train.index[-1] < df_test.index[0]:
    print("✅ Train data ends before test data begins - correct temporal split!")
else:
    print("⚠️  WARNING: Train data extends into test period!") 