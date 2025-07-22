#!/usr/bin/env python3
"""
Script to run all regional XGBoost models sequentially.
This script will execute all four regional models: Kyoto, Osaka, Tokyo, and Tsushima.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_model(model_name, script_name):
    """Run a specific regional model and handle any errors"""
    print(f"\n{'='*60}")
    print(f"Starting {model_name} model...")
    print(f"{'='*60}")
    
    try:
        # Run the model script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        print(f"✅ {model_name} model completed successfully!")
        print(f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Warnings/Info: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {model_name} model:")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {model_name} model: {e}")
        return False
    
    return True

def main():
    """Main function to run all regional models"""
    print("🚀 Starting Regional XGBoost Microplastic Prediction Models")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define the models to run
    models = [
        ("Global", "Gpower_Xgb_Main_global.py"),
        ("Kyoto", "Gpower_Xgb_Main_kyoto.py"),
        ("Osaka", "Gpower_Xgb_Main_osaka.py"),
        ("Tokyo", "Gpower_Xgb_Main_tokyo.py"),
        ("Tsushima", "Gpower_Xgb_Main_tsushima.py")
    ]
    
    # Track results
    successful_models = []
    failed_models = []
    
    # Run each model
    for model_name, script_name in models:
        if os.path.exists(script_name):
            success = run_model(model_name, script_name)
            if success:
                successful_models.append(model_name)
            else:
                failed_models.append(model_name)
        else:
            print(f"❌ Script not found: {script_name}")
            failed_models.append(model_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful models: {len(successful_models)}")
    for model in successful_models:
        print(f"   - {model}")
    
    print(f"❌ Failed models: {len(failed_models)}")
    for model in failed_models:
        print(f"   - {model}")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_models:
        print("\n⚠️  Some models failed. Check the error messages above.")
        return 1
    else:
        print("\n🎉 All models completed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 