#!/usr/bin/env python3

import pandas as pd
from datetime import datetime, timedelta

def extract_cygnss_averages():
    """Extract microplastic concentration averages from CYGNSS data"""
    
    cygnss_file = "/Users/maradumitru/Downloads/CYGNSS-data/concentration_averages_per_file.txt"
    
    # Read the averages, skipping comment lines
    averages = []
    with open(cygnss_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract just the number (remove comments)
                value = float(line.split('#')[0].strip())
                averages.append(value)
    
    print(f"Extracted {len(averages)} microplastic concentration averages")
    print(f"Range: {min(averages):.3f} to {max(averages):.3f}")
    
    return averages

def create_cygnss_user_data(averages):
    """Create user_data.py file with CYGNSS averages"""
    
    # Create the data file
    with open('user_data.py', 'w') as f:
        f.write("# CYGNSS Microplastic Concentration Averages\n")
        f.write("# One average per CYGNSS file\n")
        f.write(f"# Total averages: {len(averages)}\n\n")
        f.write("data = [\n")
        
        # Write averages in groups of 5 for readability
        for i in range(0, len(averages), 5):
            group = averages[i:i+5]
            f.write("    " + ", ".join(f"{val}" for val in group) + ",\n")
        
        f.write("]\n")
    
    print(f"Created user_data.py with {len(averages)} CYGNSS averages")

def create_cygnss_household_data(averages):
    """Create household_power_consumption.txt with CYGNSS averages"""
    
    # Start datetime 
    start_date = datetime(2018, 8, 16, 12, 0, 0)  # Starting from first CYGNSS file date
    
    # Create header
    header = "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
    
    # Open file for writing
    with open('household_power_consumption.txt', 'w') as f:
        f.write(header)
        
        # Write each data point (assume daily averages, so increment by 1 day each time)
        for i, value in enumerate(averages):
            # Calculate current datetime (increment by 1 day each time for daily averages)
            current_time = start_date + timedelta(days=i)
            
            # Format datetime
            date_str = current_time.strftime("%d/%m/%Y")
            time_str = current_time.strftime("%H:%M:%S")
            
            # Write row with CYGNSS average as Global_active_power
            # Keep other values constant to match expected format
            f.write(f"{date_str};{time_str};{value};0.4;241.0;18.4;0.0;1.0;17.0\n")
    
    print(f"Created household_power_consumption.txt with {len(averages)} CYGNSS averages")
    print(f"Time range: {start_date.date()} to {(start_date + timedelta(days=len(averages)-1)).date()}")

def main():
    print("=== CYGNSS Microplastic Concentration Data Processing ===")
    
    # Extract averages from CYGNSS data
    averages = extract_cygnss_averages()
    
    # Create user_data.py file
    create_cygnss_user_data(averages)
    
    # Create household_power_consumption.txt file  
    create_cygnss_household_data(averages)
    
    print("\n=== Data Processing Complete ===")
    print("Now you can run the prediction scripts:")
    print("1. python Gpower_Xgb_Main.py")
    print("2. python lstm_Main.py") 
    print("3. python Gpower_Arima_Main.py")

if __name__ == "__main__":
    main() 