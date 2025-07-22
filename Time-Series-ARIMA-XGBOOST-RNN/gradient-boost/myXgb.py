from util import *
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt

# get data for train, test, and forecast(unseen)


def xgb_data_split(df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols):
    # generate unseen data
    unseen = get_unseen_data(unseen_start_date, steps,
                             encode_cols, bucket_size)
    df = pd.concat([df, unseen], axis=0)
    df = date_transform(df, encode_cols)
    
    # Sort index and remove duplicates to fix slicing issues
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # data for forecast ,skip the connecting point
    df_unseen = df[unseen_start_date:].iloc[:, 1:]
    # skip the connecting point
    df_test = df[test_start_date: unseen_start_date].iloc[:-1, :]
    df_train = df[:test_start_date]
    return df_unseen, df_test, df_train


def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    # df.plot()
    df.plot(kind='barh', x='feature', y='fscore',
            legend=False, figsize=(12, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Feature importance plot saved as: {title}.png")


def xgb_importance(df, test_ratio, xgb_params, ntree, early_stop, plot_title):
    df = pd.DataFrame(df)
    # split the data into train/test set
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_ratio,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'validate')]

    xgb_model = xgb.train(xgb_params, dtrain, ntree, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=True)

    importance = xgb_model.get_fscore()
    importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
    feature_importance_plot(importance_sorted, plot_title)


def xgb_forecasts_plot(plot_start, Y, Y_test, Y_hat, forecasts, title):
    Y = pd.concat([Y, Y_test])
    ax = Y[plot_start:].plot(label='observed', figsize=(15, 10))
    #Y_test.plot(label='test_observed', ax=ax)
    Y_hat.plot(label="test_predicted", ax=ax)
    forecasts.plot(label="forecasts", ax=ax)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(Y_test.index[0]), Y_test.index[-1],
                     alpha=.1, zorder=-1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Microplastic Concentration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Forecasting plot saved as: {title}.png")

def xgb_forecasts_plot_with_actual(plot_start, Y, Y_test, Y_hat, forecasts, actual_forecast_observed, title):
    """Plot forecasts with actual observed data for the forecast period"""
    Y = pd.concat([Y, Y_test])
    ax = Y[plot_start:].plot(label='observed', figsize=(15, 10), color='blue', alpha=0.7)
    
    # Plot test predictions
    Y_hat.plot(label="test_predicted", ax=ax, color='orange', linewidth=2)
    
    # Plot forecasts
    forecasts.plot(label="forecasts", ax=ax, color='green', linewidth=2)
    
    # Plot actual observed data for forecast period (if available)
    if len(actual_forecast_observed) > 0:
        actual_forecast_observed.plot(label="actual_forecast_observed", ax=ax, color='red', linewidth=2, linestyle='--')
    
    # Highlight test period
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(Y_test.index[0]), Y_test.index[-1],
                     alpha=0.1, color='blue', zorder=-1, label='Test Period')
    
    # Highlight forecast period
    if len(actual_forecast_observed) > 0:
        ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(forecasts.index[0]), forecasts.index[-1],
                         alpha=0.1, color='green', zorder=-1, label='Forecast Period')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Microplastic Concentration')
    ax.set_title(f'{title} - Microplastic Concentration Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Forecasting plot with actual observed data saved as: {title}.png")
