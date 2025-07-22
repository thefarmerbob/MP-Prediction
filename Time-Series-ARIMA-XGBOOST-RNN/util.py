import pandas as pd
import numpy as np
from matplotlib import dates
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def preprocess(N_rows, parse_dates, filename):
    total_rows = sum(1 for l in open(filename))
    variable_names = pd.read_csv(
        filename, header=0, delimiter=';', nrows=5)
    # Read from the beginning of the file instead of the end
    # Remove skiprows parameter to read from start
    df = pd.read_csv(filename, header=0, delimiter=';', names=variable_names.columns,
                     parse_dates=parse_dates, index_col=0, nrows=N_rows)
    df_no_na = df.replace('?', np.nan)
    df_no_na.dropna(inplace=True)
    return df_no_na.astype(float)


def timeseries_plot(y, color, y_label):
    # y is Series with index of datetime
    from matplotlib.ticker import MaxNLocator
    
    # Calculate data span to choose appropriate tick spacing
    data_span_days = (y.index[-1] - y.index[0]).days
    
    # Choose tick spacing based on data span to avoid too many ticks
    if data_span_days > 365 * 2:  # More than 2 years
        # Use monthly major ticks and yearly minor ticks for large datasets
        major_locator = dates.MonthLocator(interval=3)  # Every 3 months
        major_formatter = dates.DateFormatter('%Y-%m')
        minor_locator = dates.YearLocator()
        minor_formatter = dates.DateFormatter('')
    elif data_span_days > 365:  # 1-2 years
        # Use monthly ticks
        major_locator = dates.MonthLocator(interval=2)  # Every 2 months  
        major_formatter = dates.DateFormatter('%Y-%m')
        minor_locator = dates.MonthLocator()
        minor_formatter = dates.DateFormatter('%m')
    elif data_span_days > 90:  # 3-12 months
        # Use weekly ticks
        major_locator = dates.WeekdayLocator(interval=2)  # Every 2 weeks
        major_formatter = dates.DateFormatter('%m-%d')
        minor_locator = dates.WeekdayLocator()
        minor_formatter = dates.DateFormatter('%d')
    else:  # Less than 3 months
        # Use original daily ticks for small datasets
        major_locator = dates.WeekdayLocator(byweekday=(), interval=1)
        major_formatter = dates.DateFormatter('\n\n%a')
        minor_locator = dates.DayLocator()
        minor_formatter = dates.DateFormatter('%m-%d')

    fig, ax = plt.subplots()
    
    # Set locators with adaptive spacing to prevent tick overflow
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_minor_formatter(minor_formatter)

    ax.set_ylabel(y_label)
    ax.plot(y.index, y, color)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(y_label + '.png', dpi=300)
    plt.show()

# average time series


def bucket_avg(ts, bucket):
    # ts is Sereis with index
    # bucket =["30min","60min","M".....]
    # Convert old 'T' format to new 'min' format
    if bucket.endswith('T'):
        bucket = bucket[:-1] + 'min'
    y = ts.resample(bucket).mean()
    return y


def config_plot():
    plt.style.use('seaborn-v0_8-paper')
#    plt.rcParams.update({'axes.prop_cycle': cycler(color='jet')})
    plt.rcParams.update({'axes.titlesize': 20})
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 16})
    plt.rcParams.update({'ytick.labelsize': 16})
    plt.rcParams.update({'figure.figsize': (10, 6)})
    plt.rcParams.update({'legend.fontsize': 20})
    return 1


# static xgboost
# get one-hot encoder for features
def date_transform(df, encode_cols):
    # extract a few features from datetime
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['WeekofYear'] = df.index.isocalendar().week
    df['DayofWeek'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    # one hot encoder for categorical variables
    for col in encode_cols:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, columns=encode_cols)
    return df


def get_unseen_data(unseen_start, steps, encode_cols, bucket_size):
    # Convert old 'T' format to new 'min' format
    if bucket_size.endswith('T'):
        bucket_size = bucket_size[:-1] + 'min'
    index = pd.date_range(unseen_start,
                          periods=steps, freq=bucket_size)
    df = pd.DataFrame(pd.Series(np.zeros(steps), index=index),
                      columns=['Global_active_power'])
    return df

# dynamic xgboost
# shift 2 steps for every lag


def data_add_timesteps(data, column, lag):
    column = data[column]
    step_columns = [column.shift(i) for i in range(2, lag + 1, 2)]
    df_steps = pd.concat(step_columns, axis=1)
    # current Global_active_power is at first columns
    df = pd.concat([data, df_steps], axis=1)
    return df
