"""
Optuna-Optimized XGBoost Parameters Configuration
=================================================

This module provides easy access to the optimized XGBoost parameters found through
Optuna hyperparameter optimization (50 trials, best RMSE: 2.111101).

Usage Examples:
    from optuna_params_config import get_optimized_params, apply_to_model
    
    # Get the optimized parameters
    params = get_optimized_params()
    
    # Use with native XGBoost
    import xgboost as xgb
    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
    
    # Use with sklearn wrapper
    from xgboost import XGBRegressor
    model = apply_to_model(XGBRegressor())
"""

import pandas as pd
from datetime import datetime

# Optuna optimization results
OPTIMIZATION_RESULTS = {
    'optimization_date': '2025-07-22',
    'n_trials': 50,
    'best_rmse': 2.111101,
    'best_trial_number': 45,
    'optimization_time_minutes': 15,
    'optimized_parameters': ['learning_rate', 'max_depth', 'n_estimators'],
    'fixed_parameters': ['subsample', 'colsample_bytree', 'min_child_weight', 
                        'reg_alpha', 'reg_lambda', 'gamma']
}

def get_optimized_params(format='native'):
    """
    Get the Optuna-optimized XGBoost parameters.
    
    Args:
        format (str): 'native' for xgb.train(), 'sklearn' for XGBRegressor()
        
    Returns:
        dict: Optimized parameters dictionary
    """
    # Base optimized parameters (for xgb.train)
    native_params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'subsample': 0.58,
        'colsample_bytree': 0.499,
        'min_child_weight': 14,
        'learning_rate': 0.03766600333324149,  # Optuna-optimized
        'max_depth': 6,  # Optuna-optimized
        'n_estimators': 900,  # Optuna-optimized
        'reg_alpha': 27.12,
        'reg_lambda': 10.34,
        'gamma': 2.24,
        'tree_method': 'auto',
        'sampling_method': 'uniform',
        'seed': 42
    }
    
    if format == 'native':
        return native_params
    elif format == 'sklearn':
        # Convert to sklearn XGBRegressor format
        sklearn_params = native_params.copy()
        sklearn_params.pop('n_estimators')  # handled separately in sklearn
        sklearn_params.pop('sampling_method')  # not used in sklearn
        sklearn_params.pop('booster')  # default in sklearn
        return sklearn_params
    else:
        raise ValueError("format must be 'native' or 'sklearn'")

def apply_to_model(model_class, **kwargs):
    """
    Apply optimized parameters to an XGBoost model.
    
    Args:
        model_class: XGBRegressor class or instance
        **kwargs: Additional parameters to override
        
    Returns:
        Configured XGBoost model instance
    """
    from xgboost import XGBRegressor
    
    if isinstance(model_class, type):
        # It's a class, create instance
        optimized_params = get_optimized_params('sklearn')
        optimized_params.update(kwargs)
        return model_class(n_estimators=900, **optimized_params)
    else:
        # It's an instance, update parameters
        optimized_params = get_optimized_params('sklearn')
        optimized_params.update(kwargs)
        model_class.set_params(n_estimators=900, **optimized_params)
        return model_class

def get_improvement_summary():
    """
    Get a summary of improvements from the original parameters.
    
    Returns:
        dict: Summary of parameter improvements
    """
    return {
        'learning_rate': {
            'original': 0.09,
            'optimized': 0.03766600333324149,
            'improvement': 'Reduced by 58.1% for better convergence'
        },
        'max_depth': {
            'original': 3,
            'optimized': 6,
            'improvement': 'Increased by 100% for better model complexity'
        },
        'n_estimators': {
            'original': 850,
            'optimized': 900,
            'improvement': 'Increased by 5.9% for better performance'
        },
        'validation_rmse': {
            'improvement': 'Achieved 2.111101 RMSE (excellent performance)'
        }
    }

def print_optimization_summary():
    """Print a formatted summary of the optimization results."""
    print("=" * 60)
    print("🎯 OPTUNA XGBOOST OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"📅 Optimization Date: {OPTIMIZATION_RESULTS['optimization_date']}")
    print(f"🔬 Number of Trials: {OPTIMIZATION_RESULTS['n_trials']}")
    print(f"🏆 Best RMSE: {OPTIMIZATION_RESULTS['best_rmse']:.6f}")
    print(f"⏱️  Optimization Time: ~{OPTIMIZATION_RESULTS['optimization_time_minutes']} minutes")
    print()
    
    print("📈 PARAMETER IMPROVEMENTS:")
    improvements = get_improvement_summary()
    for param, info in improvements.items():
        if param != 'validation_rmse':
            print(f"  • {param}:")
            print(f"    Original: {info['original']}")
            print(f"    Optimized: {info['optimized']}")
            print(f"    {info['improvement']}")
    print(f"  • {improvements['validation_rmse']['improvement']}")
    print()
    
    print("✅ READY TO USE:")
    print("  from optuna_params_config import get_optimized_params, apply_to_model")
    print("  params = get_optimized_params()  # For xgb.train()")
    print("  model = apply_to_model(XGBRegressor())  # For sklearn")

def save_params_to_file(filename=None):
    """
    Save optimized parameters to a file for external use.
    
    Args:
        filename (str): Output filename. If None, uses timestamp.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xgb_optimized_params_{timestamp}.txt"
    
    params = get_optimized_params('native')
    sklearn_params = get_optimized_params('sklearn')
    
    with open(filename, 'w') as f:
        f.write("# Optuna-Optimized XGBoost Parameters\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Best RMSE: {OPTIMIZATION_RESULTS['best_rmse']:.6f}\n")
        f.write(f"# Trials: {OPTIMIZATION_RESULTS['n_trials']}\n\n")
        
        f.write("# For xgb.train() - Native XGBoost\n")
        f.write("native_params = {\n")
        for key, value in params.items():
            if isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            else:
                f.write(f"    '{key}': {value},\n")
        f.write("}\n\n")
        
        f.write("# For XGBRegressor() - Sklearn Wrapper\n")
        f.write("sklearn_params = {\n")
        for key, value in sklearn_params.items():
            if isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            else:
                f.write(f"    '{key}': {value},\n")
        f.write("}\n")
        f.write("n_estimators = 900\n")
    
    print(f"✅ Parameters saved to: {filename}")

if __name__ == "__main__":
    # Print summary when run directly
    print_optimization_summary()
    
    # Save parameters to file
    save_params_to_file()
    
    # Example usage
    print("\n" + "=" * 60)
    print("📝 EXAMPLE USAGE:")
    print("=" * 60)
    
    # Show native usage
    print("\n🔹 Native XGBoost Usage:")
    print("import xgboost as xgb")
    print("from optuna_params_config import get_optimized_params")
    print("")
    print("params = get_optimized_params('native')")
    print("model = xgb.train(params, dtrain, num_boost_round=900)")
    
    # Show sklearn usage  
    print("\n🔹 Sklearn XGBRegressor Usage:")
    print("from xgboost import XGBRegressor") 
    print("from optuna_params_config import apply_to_model")
    print("")
    print("model = apply_to_model(XGBRegressor)")
    print("model.fit(X_train, y_train)")
    
    print("\n🎉 Ready to use optimized parameters in your models!") 