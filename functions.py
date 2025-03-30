import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def prepare_stock_data(csv_path, start_date=None, end_date=None, feature_engineering=True, standardize=True, plot=False, threshold = 0.7):
    """
    Clean and prepare stock data from a CSV file for use in a DQN trading model.
    
    Parameters:
    csv_path (str): Path to the CSV file containing stock data
    start_date (str, optional): Start date in 'YYYY-MM-DD' format
    end_date (str, optional): End date in 'YYYY-MM-DD' format
    feature_engineering (bool): Whether to create engineered features like returns and deltas
    standardize (bool): Whether to apply min-max standardization to the state space
    
    Returns:
    tuple: (df_raw, df_state, state_columns, scaler)
        - df_raw: Raw dataframe with price information (for reward calculation)
        - df_state: Processed dataframe with state features (standardized if requested)
        - state_columns: List of column names used in the state space
        - scaler: Fitted scaler object (if standardization was applied)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Initial data shape: {df.shape}")
    required_columns = ['high', 'low', 'open', 'close']
    for col in required_columns:
        if col not in df.columns and col.capitalize() not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
    
    if 'date' not in df.columns:
        date_col = [col for col in df.columns if 'date' in col.lower()]
        if date_col:
            df.rename(columns={date_col[0]: 'date'}, inplace=True)
        else:
            raise ValueError("No date column found in the CSV file")
        
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    

    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values:")
        print(missing_values[missing_values > 0])
        df.fillna(method='ffill', inplace=True)
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            df.fillna(method='bfill', inplace=True)
        print(f"After filling, {df.isnull().sum().sum()} missing values remain")
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    # Store raw dataframe for calculating rewards
    df_raw = df.copy()
    
    # Feature engineering as described in https://medium.com/@adamdarmanin/deep-q-learning-applied-to-algorithmic-trading-c91068791d68
    if feature_engineering:
        print("\nEngineering features...")
        
        # 1. Price Returns
        df['return_close'] = df['close'].pct_change()
        df['return_open'] = df['open'].pct_change() 
        df['return_high'] = df['high'].pct_change()
        df['return_low'] = df['low'].pct_change()
        
        # 2. Price Delta
        df['delta'] = df['high'] - df['low']
        
        # 3. Normalized Delta
        df['norm_delta'] = df['delta'] / df['open']
        
        # 4. Moving averages
        df['ma_5d'] = df['close'].rolling(window=5).mean()
        df['ma_20d'] = df['close'].rolling(window=20).mean()
        
        # 5. Volatility (20-day rolling standard deviation of returns)
        df['volatility_20d'] = df['return_close'].rolling(window=20).std()
        
        # 6. MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 7. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 8. Volume features
        if 'volume' in df.columns:
            df['volume_return'] = df['volume'].pct_change()
            df['volume_ma_5d'] = df['volume'].rolling(window=5).mean()
            df['volume_ratio_5d'] = df['volume'] / df['volume_ma_5d']
    
    # Remove rows with NaN values created by the rolling windows and pct_change
    df_clean = df.dropna()
    
    # Define state space columns
    state_columns = [
        'return_close', 'return_open', 'return_high', 'return_low',
        'delta', 'norm_delta', 'ma_5d', 'ma_20d', 'volatility_20d',
        'macd', 'macd_signal', 'rsi', 'volume_return', 'volume_ma_5d', 'volume_ratio_5d'
    ]
    
    df_state = df_clean[state_columns].copy()
    
    scaler = None
    if standardize:
        print("\nApplying min-max standardization to state features...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_state_scaled = pd.DataFrame(
            scaler.fit_transform(df_state),
            columns=df_state.columns,
            index=df_state.index
        )
        df_state = df_state_scaled
    
    print(f"\nData period: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
    print(f"Number of trading days: {len(df_clean)}")
    
    print("\nKey Statistics:")
    print(f"Average Daily Return: {df_clean['return_close'].mean() * 100:.4f}%")
    print(f"Return Volatility: {df_clean['return_close'].std() * 100:.4f}%")
    total_return = ((df_clean['close'].iloc[-1] / df_clean['close'].iloc[0]) - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    days = (df_clean.index.max() - df_clean.index.min()).days
    years = days / 365.25
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Sharpe Ratio (assuming risk-free rate of 0): {(annualized_return / (df_clean['return_close'].std() * np.sqrt(252) * 100)):.4f}")

    correlation_matrix = df_state.corr()
    correlation_matrix_abs = correlation_matrix.abs()
    strong_correlations = (correlation_matrix_abs >= threshold) & (correlation_matrix_abs < 1)
    strong_pairs = set()
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if strong_correlations.loc[i, j]:
                if i < j:
                    strong_pairs.add((i, j))
    print("\nStrong Correlations above threshold:")
    print(strong_pairs)

    # Remove features over teh threshold for the correlation matrix
    columns_to_remove = set()
    for i, j in strong_pairs:
        if j not in columns_to_remove:
            columns_to_remove.add(j)
    print(columns_to_remove)
    df_state = df_state.drop(columns=columns_to_remove)
    state_columns = [col for col in state_columns if col not in columns_to_remove]

    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar=False)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(df_clean.index, df_clean['close'])
        plt.title('Stock Price Over Time')
        plt.ylabel('Price')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df_clean.index, df_clean['return_close'] * 100)
        plt.title('Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.tight_layout()
    
    return df_raw, df_state, state_columns, scaler

def visualize_state_features(df_state, n_features=None):
    """
    Visualize the engineered state features.
    
    Parameters:
    df_state (pd.DataFrame): DataFrame containing the state features
    n_features (int, optional): Number of features to plot. If None, plot all.
    """
    features = df_state.columns[:n_features] if n_features else df_state.columns
    n_cols = 2
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(df_state.index, df_state[feature])
        plt.title(feature)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    data_raw, data_state, state_cols, scaler = prepare_stock_data(
        './dataset/individual_companies/AAPL_data.csv', 
        start_date='2014-01-01', 
        end_date='2015-12-31',
        plot=True
    )
    
    print("\nFirst 5 rows of state features:")
    print(data_state.head())
    
    print("\nState space dimensionality:", data_state.shape[1])
    print("State feature columns:", state_cols)
    
    visualize_state_features(data_state, n_features=6)
    
    plt.show()

import matplotlib.pyplot as plt

def plot_training_results(rewards, epsilons, action_counts, cumulative_rewards):
    episodes = range(len(rewards))

    # First Figure: Reward, Epsilon Decay, and Action Distribution
    plt.figure(figsize=(12, 4))

    # Reward Plot
    plt.subplot(1, 3, 1)
    plt.plot(episodes, rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()

    # Epsilon Decay
    plt.subplot(1, 3, 2)
    plt.plot(episodes, epsilons, label="Epsilon Decay", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Time")
    plt.legend()

    # Action Distribution
    plt.subplot(1, 3, 3)
    plt.bar(["Buy", "Sell", "Hold"], action_counts, color=["green", "red", "blue"])
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Selection Distribution")

    plt.tight_layout()
    plt.show()  # Show the first figure separately

    # Separate Figure for Cumulative Performance
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards, label="Cumulative Reward", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Performance Over Time")
    plt.legend()
    plt.grid(True)  
    plt.show()