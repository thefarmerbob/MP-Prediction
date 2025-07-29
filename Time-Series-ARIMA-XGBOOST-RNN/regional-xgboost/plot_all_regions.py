import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
sys.path.append('..')  # Add parent directory to path
sys.path.append('.')   # Add current directory to path
from util import *

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

def preprocess_global_data(filename):
    """Simple preprocessing function for global data format"""
    # Read the CSV file
    df = pd.read_csv(filename, delimiter=';', header=0)
    
    # Combine Date and Time columns
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('DateTime', inplace=True)
    
    # Select only the microplastic concentration column
    df = df[['Microplastic_Concentration']]
    
    return df

print("=== PLOTTING ALL REGIONS TOGETHER ===")

# Configuration
bucket_size = "1D"

# Define all regions and their details
regions = {
    'Global': {
        'filename': '../mp-avg/mp-global-avg.txt',
        'color': '#FFC0CB',  # Pink
        'preprocess_func': 'global'
    },
    'Japan': {
        'filename': '../regional-averages/mp-japan-avg.txt',
        'color': '#0000FF',  # Blue
        'preprocess_func': 'regional'
    },
    'Kyoto': {
        'filename': '../regional-averages/mp-kyoto-avg.txt',
        'color': '#00FFFF',  # Cyan
        'preprocess_func': 'regional'
    },
    'Osaka': {
        'filename': '../regional-averages/mp-osaka-avg.txt',
        'color': '#FFA500',  # Orange
        'preprocess_func': 'regional'
    },
    'Tokyo': {
        'filename': '../regional-averages/mp-tokyo-avg.txt',
        'color': '#000000',  # Black
        'preprocess_func': 'regional'
    },
    'Tsushima': {
        'filename': '../regional-averages/mp-tsushima-avg.txt',
        'color': '#00FF00',  # Lime
        'preprocess_func': 'regional'
    }
}

# Create the plot
plt.figure(figsize=(20, 12))
ax = plt.gca()

# Plot each region
for region_name, region_info in regions.items():
    print(f"Loading {region_name} data...")
    
    try:
        # Load and preprocess data based on type
        if region_info['preprocess_func'] == 'global':
            # Use the simple global preprocessing
            df = preprocess_global_data(region_info['filename'])
        else:
            # Use regional preprocessing
            df = preprocess_regional_data(region_info['filename'])
        
        # Apply bucket averaging
        mp_concentration = df["Microplastic_Concentration"]
        df_processed = pd.DataFrame(bucket_avg(mp_concentration, bucket_size))
        df_processed.dropna(inplace=True)
        
        # Plot the data
        df_processed.plot(
            ax=ax,
            label=f'{region_name} (n={len(df_processed)})',
            color=region_info['color'],
            alpha=1,
            linewidth=3
        )
        
        print(f"✅ {region_name}: {len(df_processed)} points, range: {df_processed.index[0]} to {df_processed.index[-1]}")
        
    except Exception as e:
        print(f"❌ Error loading {region_name}: {e}")
        continue

# Customize the plot
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Microplastic Concentration', fontsize=14)
ax.set_title('All Regional and Global Microplastic Concentrations', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.5)

# Improve readability
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('All_Regions_Combined_Plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✅ Combined plot saved as 'All_Regions_Combined_Plot.png'")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print("Region\t\tData Points\tMean\t\tStd\t\tMin\t\tMax")
print("-" * 80)

for region_name, region_info in regions.items():
    try:
        # Reload data for statistics
        if region_info['preprocess_func'] == 'global':
            df = preprocess_global_data(region_info['filename'])
        else:
            df = preprocess_regional_data(region_info['filename'])
        
        mp_concentration = df["Microplastic_Concentration"]
        df_processed = pd.DataFrame(bucket_avg(mp_concentration, bucket_size))
        df_processed.dropna(inplace=True)
        
        values = df_processed.iloc[:, 0]
        print(f"{region_name:<12}\t{len(values)}\t\t{values.mean():.2f}\t\t{values.std():.2f}\t\t{values.min():.2f}\t\t{values.max():.2f}")
        
    except Exception as e:
        print(f"{region_name:<12}\tError: {e}")

print("\n=== PLOT COMPLETE ===") 