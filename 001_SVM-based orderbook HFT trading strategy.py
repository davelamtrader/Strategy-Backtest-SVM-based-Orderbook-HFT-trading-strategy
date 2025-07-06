import pandas as pd
from binance.client import Client

###Step 1: Data Preparation (BTCUSDT)
# This is a conceptual function. You would need to run a WebSocket client
# or query historical tick data to build a comprehensive dataset.
def fetch_btcusdt_tick_data(api_key, api_secret, start_str, end_str):
    """
    Fetches historical aggregate trade and order book data for BTCUSDT.
    This is a simplified example. Real HFT requires a much faster data feed (e.g., WebSocket).
    
    Args:
        api_key (str): Your Binance API key.
        api_secret (str): Your Binance API secret.
        start_str (str): Start date string e.g., "1 day ago UTC".
        end_str (str): End date string e.g., "now UTC".

    Returns:
        pandas.DataFrame: A DataFrame with tick data.
    """
    client = Client(api_key, api_secret)

    # Fetching aggregate trades as a proxy for tick-by-tick trade data
    agg_trades = client.get_aggregate_trades(symbol='BTCUSDT', startTime=start_str, endTime=end_str)
    
    # This data lacks order book info. A real implementation would subscribe
    # to a WebSocket stream for combined trade and book ticker data.
    # For this example, we'll create a mock DataFrame structure that a real
    # WebSocket feed would provide.
    
    # MOCK DATA: The following simulates the data structure we need.
    # In a real scenario, you must populate this from a live WebSocket feed.
    data = {
        'timestamp': pd.to_datetime(range(1000), unit='ms'),
        'last_price': [99000 + i*0.01 + (0.1 * (-1)**i) for i in range(1000)],
        'trade_volume': [0.1 + i*0.001 for i in range(1000)],
        'best_bid': [98999.99 + i*0.01 for i in range(1000)],
        'bid_qty': [1.5 - i*0.001 for i in range(1000)],
        'best_ask': [99000.01 + i*0.01 for i in range(1000)],
        'ask_qty': [1.2 + i*0.001 for i in range(1000)],
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print("Sample Data Head:")
    print(df.head())
    
    return df

# Replace with your actual API keys and desired time range
# api_key = "YOUR_API_KEY"
# api_secret = "YOUR_API_SECRET"
# df_raw = fetch_btcusdt_tick_data(api_key, api_secret, "1 hour ago UTC", "now UTC")

# For demonstration, we use the mock data generated within the function.
df_raw = fetch_btcusdt_tick_data(None, None, None, None)

####Step 2 - Feature Engineering
import numpy as np

def create_features(df):
    """
    Engineers features based on the Minsheng Securities report.
    """
    # Basic order book features
    df['spread'] = df['best_ask'] - df['best_bid']
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    
    # Feature 5 & 6: Log returns of best prices
    df['bid_log_return'] = np.log(df['best_bid'] / df['best_bid'].shift(1))
    df['ask_log_return'] = np.log(df['best_ask'] / df['best_ask'].shift(1))
    
    # Feature 7: Relative spread [T34](10)
    df['relative_spread'] = df['spread'] / df['mid_price']
    
    # Feature 8 & 9: Log differences of best volumes [T35](11)
    df['bid_qty_log_diff'] = np.log(df['bid_qty'] / df['bid_qty'].shift(1))
    df['ask_qty_log_diff'] = np.log(df['ask_qty'] / df['ask_qty'].shift(1))
    
    # Feature 11: Depth [T36](12)
    df['depth'] = (df['bid_qty'] + df['ask_qty']) / 2
    
    # Feature 10: Slope (Spread / Depth)
    df['slope'] = df['spread'] / df['depth']
    
    # Feature 15: Cumulative volume (already given as trade_volume per tick in our data) [T38](13)
    # We will use the log difference of tick volume instead
    df['trade_vol_log_diff'] = np.log(df['trade_volume'] / df['trade_volume'].shift(1))

    # Add other common technical indicators as suggested by the report [T1](5)
    df['mid_price_ma_5'] = df['mid_price'].rolling(window=5).mean()
    df['mid_price_ma_10'] = df['mid_price'].rolling(window=10).mean()

    # The report uses 17 features. We implemented the applicable ones.
    # Features 12, 13 (Open Interest) and 17 (Basis) are omitted for Spot.
    # The final list of features for the model:
    feature_list = [
        'spread', 'mid_price', 'bid_log_return', 'ask_log_return', 
        'relative_spread', 'bid_qty_log_diff', 'ask_qty_log_diff', 
        'depth', 'slope', 'trade_vol_log_diff', 'mid_price_ma_5', 'mid_price_ma_10'
    ]
    
    # Drop rows with NaN values created by shifts and rolling windows
    df_features = df.dropna()
    
    return df_features, feature_list


df_features, feature_names = create_features(df_raw.copy())
print("\nFeatures DataFrame Head:")
print(df_features[feature_names].head())

###Step 3: Define Trading Rule (Target Variable)
def create_target(df, delta_t=2, price_threshold=0.05):
    """
    Creates the target variable for the classification model.
    
    Args:
        df (pandas.DataFrame): DataFrame with mid_price.
        delta_t (int): The number of future ticks to predict over.
        price_threshold (float): The minimum mid-price change to be considered a non-flat move.

    Returns:
        pandas.Series: The target variable series.
    """
    future_mid_price = df['mid_price'].shift(-delta_t)
    price_change = future_mid_price - df['mid_price']
    
    df['target'] = 0
    df.loc[price_change > price_threshold, 'target'] = 1   # "Up"
    df.loc[price_change < -price_threshold, 'target'] = -1  # "Down"
    
    return df.dropna()

# delta_t=2 ticks is used, similar to the report's analysis [T41](8)
# price_threshold for BTCUSDT needs to be determined from data analysis; 0.05 is an example.
df_model_data = create_target(df_features.copy(), delta_t=2, price_threshold=0.05)
print("\nData with Target Variable:")
print(df_model_data[['mid_price', 'target']].head())
print("\nTarget Distribution:")
print(df_model_data['target'].value_counts())


###Step 4 & 5: SVM Model Training and Hyperparameter Tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Prepare data for the model
# Filter out "flat" cases to focus on predicting significant moves
df_filtered = df_model_data[df_model_data['target'] != 0].copy()

X = df_filtered[feature_names]
y = df_filtered['target']

# 2. Split data chronologically to prevent look-ahead bias
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Hyperparameter Tuning using GridSearchCV
# These parameters are a starting point; a real application would test a wider range.
param_grid = {
    'C': [0.1, 1, 10], 
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf'] # Radial Basis Function is good for non-linear problems
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_svm = grid_search.best_estimator_
print(f"\nBest SVM Parameters: {grid_search.best_params_}")

# 5. Train final model with best parameters
# GridSearchCV automatically retrains the best model on the whole training set.

###Step 6: SVM Model Testing and Refining
# 1. Make predictions on the test set
y_pred = best_svm.predict(X_test_scaled)

# 2. Evaluate model performance
# The report shows test accuracies around 60-70% [T2](2)
print("\n--- Model Performance on Test Set ---")
print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Up (1)']))

# 3. Simple Backtest Simulation
def run_backtest(X_test, y_test, y_pred, delta_t=2, commission=0.0004, slippage=0.01):
    """
    Runs a simple vectorized backtest.
    
    Args:
        X_test (DataFrame): Test features.
        y_test (Series): True labels.
        y_pred (array): Predicted labels.
        delta_t (int): Holding period in ticks.
        commission (float): Transaction fee per trade.
        slippage (float): Estimated slippage per trade in price points.
    """
    backtest_df = X_test.copy()
    backtest_df['prediction'] = y_pred
    backtest_df['pnl'] = 0.0

    # Calculate PnL for each trade
    for i in range(len(backtest_df) - delta_t):
        signal = backtest_df['prediction'].iloc[i]
        
        # We use future prices from the data to simulate closing the trade
        # This is a simplification; a real backtester is more complex.
        entry_price = 0
        exit_price = 0
        
        if signal == 1: # Predicted UP -> Go Long
            entry_price = backtest_df['best_ask'].iloc[i] + slippage
            exit_price = backtest_df['best_bid'].iloc[i + delta_t] - slippage
        elif signal == -1: # Predicted DOWN -> Go Short
            entry_price = backtest_df['best_bid'].iloc[i] - slippage
            exit_price = backtest_df['best_ask'].iloc[i + delta_t] + slippage

        if entry_price > 0:
            pnl = (exit_price - entry_price) * signal
            pnl -= (entry_price + exit_price) * commission # Commission on entry and exit
            backtest_df['pnl'].iloc[i] = pnl
    
    # Calculate summary statistics
    trades = backtest_df[backtest_df['prediction'] != 0]
    winning_trades = trades[trades['pnl'] > 0]
    
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    net_profit = trades['pnl'].sum()
    
    print("\n--- Backtest Simulation Results ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}") # Report achieved a 56% win rate [T2](2)
    print(f"Net Profit: {net_profit:.2f}") # Report achieved a net profit of 11814.99 yuan in a day [T2](2)

run_backtest(X_test, y_test, y_pred)


























