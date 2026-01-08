import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

def compute_indicators(df, close_col='Close'):
    df = df.copy().sort_values('Date')
    price = df[close_col]

    # Moving averages
    df['SMA_50'] = price.rolling(50, min_periods=1).mean()
    df['SMA_200'] = price.rolling(200, min_periods=1).mean()

    # Exponential MAs
    df['EMA_12'] = price.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = price.ewm(span=26, adjust=False).mean()

    # MACD and signal
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # RSI (14)
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(50)

    # Bollinger Bands (20,2)
    sma20 = price.rolling(20, min_periods=1).mean()
    std20 = price.rolling(20, min_periods=1).std().fillna(0)
    df['BB_mid'] = sma20
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20

    # ATR (14) using high/low/close if present; fall back to pct change
    if {'High','Low','Close'}.issubset(df.columns):
        high = df['High']; low = df['Low']; close = df['Close']
        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14, min_periods=1).mean()
    else:
        df['ATR_14'] = price.pct_change().abs().rolling(14, min_periods=1).mean() * price

    # # Drawdown from running max
    # df['RollingMax'] = price.cummax()
    # df['Drawdown'] = price / df['RollingMax'] - 1

    return df

def download_ticker(ticker, start="2019-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None
    df = df.reset_index()
    df['Ticker'] = ticker
    return df

def download_multiple_tickers(ticker_source, start="2019-01-01", end=None):
    """
    Downloads historical data for multiple tickers.
    ticker_source: Can be a list of strings OR a path to a CSV file.
    """
    
    # 1. Handle CSV input automatically
    if isinstance(ticker_source, str) and ticker_source.endswith('.csv'):
        try:
            # Assuming the CSV has a column named 'Ticker' or is just a list
            df_csv = pd.read_csv(ticker_source)
            if 'Ticker' in df_csv.columns:
                ticker_list = df_csv['Ticker'].tolist()
            else:
                # Fallback: if no 'Ticker' column, assume the first column contains tickers
                ticker_list = df_csv.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error reading CSV {ticker_source}: {e}")
            return {}
    else:
        # If it's already a list, use it directly
        ticker_list = ticker_source

    # 2. Helper function (unchanged)
    def download_ticker(ticker, start=start, end=end):
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"Warning: Could not download data for {ticker}. Skipping.")
            return None
        df = df.reset_index()
        df['Ticker'] = ticker
        return df

    # 3. Main execution logic
    all_data = {
        ticker: download_ticker(str(ticker).strip()) 
        for ticker in ticker_list
        if ticker is not None
    }
    
    # Filter out None results efficiently
    all_data = {k: v for k, v in all_data.items() if v is not None}
    
    return all_data

def calculate_signal_probabilities(all_data):
    """
    Iterates through a dictionary of DataFrames, renames columns, calculates 
    technical indicators, derives buy/risk factors, and computes weighted 
    Sell_Probability and Buy_Probability.

    Args:
        all_data (dict): A dictionary where keys are tickers and values are 
                         their corresponding downloaded DataFrames.

    Returns:
        dict: A dictionary of DataFrames with all new indicator and probability columns.
    """
    
    results = {}
    
    # NOTE: This column list is dependent on the EXACT structure of your input DataFrames.
    correct_columns = [
        'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 
        'Ticker'
    ]   

    for t, df in all_data.items():
        # --- Step 1: Column Standardization ---
        # Assuming the downloaded DataFrame columns need to be renamed/reordered
        if len(df.columns) == len(correct_columns):
            df.columns = correct_columns
        
        # --- Step 2: Calculate Technical Indicators ---
        # NOTE: This requires the 'compute_indicators' function to be defined/imported.
        df = compute_indicators(df)
        
        # --- Step 3: Derive Risk Factors ---
        
        # Calculate max MACD_hist for normalization, adding a zero check
        macd_hist_max = df['MACD_hist'].abs().max()
        
        df['RSI_risk'] = ((df['RSI_14'] - 50) / 50).clip(0, 1)
        
        # Use a ternary operator to prevent division by zero
        df['MACD_risk'] = (-df['MACD_hist'] / macd_hist_max).clip(0, 1) if macd_hist_max != 0 else 0
        
        df['Trend_risk'] = ((df['SMA_200'] - df['Close']) / df['SMA_200']).clip(0, 1)
        df['BB_pos'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_risk'] = df['BB_pos'].clip(0, 1)
        df['ATR_risk'] = (df['ATR_14'] / df['Close']).clip(0, 1)
        
        # --- Step 4: Derive Buy Factors ---
        
        df['RSI_buy'] = ((30 - df['RSI_14']) / 30).clip(0, 1)
        
        # Use a ternary operator to prevent division by zero
        df['MACD_buy'] = (df['MACD_hist'] / macd_hist_max).clip(0, 1) if macd_hist_max != 0 else 0
        
        df['Trend_buy'] = ((df['Close'] - df['SMA_50']) / df['Close']).clip(0, 1)
        df['BB_buy'] = (1 - df['BB_pos']).clip(0, 1)
        df['ATR_buy'] = (1 - (df['ATR_14'] / df['Close'])).clip(0, 1)

        # --- Step 5: Calculate Weighted Probabilities ---

        df['Sell_Probability'] = (
              0.30 * df['Trend_risk']
            + 0.25 * df['RSI_risk']
            + 0.20 * df['MACD_risk']
            + 0.15 * df['BB_risk']
            + 0.10 * df['ATR_risk']
        ).clip(0, 1)

        df['Buy_Probability'] = (
              0.25 * df['RSI_buy']
            + 0.25 * df['MACD_buy']
            + 0.20 * df['Trend_buy']
            + 0.15 * df['BB_buy']
            + 0.10 * df['ATR_buy']
        ).clip(0, 1)

        # Store a copy of the final processed DataFrame
        results[t] = df.copy() 
        
    return results

def create_summary_table(results):
    """
    Generates a summary DataFrame of the latest close price and 
    calculated Buy/Sell probabilities for all tickers.

    Args:
        results (dict): A dictionary where keys are tickers and values are 
                        the processed DataFrames containing probability columns.

    Returns:
        pd.DataFrame: A summary table with latest metrics for each ticker.
    """
    
    # Use a list comprehension to build the list of summary dictionaries
    summary_data = [
        {
            "Ticker": t,
            "Latest_Close": df['Close'].iloc[-1],
            "Latest_Sell_Prob": df['Sell_Probability'].iloc[-1],
            "Latest_Buy_Prob": df['Buy_Probability'].iloc[-1],
            # Calculate the absolute difference between the probabilities
            "Latest_Diff_Prob": abs(df['Buy_Probability'].iloc[-1] - df['Sell_Probability'].iloc[-1]),
        }
        # Iterate over the items in the results dictionary
        for t, df in results.items()
    ]
    
    # Create the final DataFrame from the list of dictionaries
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df


import plotly.graph_objects as go
# Assuming results is the dictionary containing processed DataFrames
# with 'Close' and 'Sell_Probability' columns

import plotly.graph_objects as go

def plot_interactive_signals(results, probability_column='Buy_Probability'):
    """
    Generates a Plotly figure with a dropdown menu to dynamically switch 
    between tickers, showing Price vs. a specified Probability (Buy or Sell).
    
    Args:
        results (dict): Dictionary of processed DataFrames.
        probability_column (str): The name of the probability column to plot,
                                  e.g., 'Buy_Probability' or 'Sell_Probability'.
        
    Returns:
        go.Figure: The interactive Plotly figure object.
    """
    
    # Dynamically determine the short name for the legend/title
    prob_short_name = probability_column.replace('_Probability', ' Strength')
    
    fig = go.Figure()

    buttons = []
    num = 0
    first = True
    
    total_traces = 2 * len(results)
    
    for t, df in results.items():
        visible_state = True if first else False

        # --- TRACE 1: Close Price ---
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'], visible=visible_state,
            name=f'{t} Close', hoverinfo='x+y',
            yaxis='y1'
        ))
        
        # --- TRACE 2: Probability Trace (Dynamically set based on input) ---
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[probability_column], visible=visible_state,
            name=f'{t} {prob_short_name}', yaxis='y2', hoverinfo='x+y'
        ))

        # --- Create Button Visibility List ---
        visible_list = [False] * total_traces
        vis_index = 2 * num
        visible_list[vis_index] = True      # Price trace visible
        visible_list[vis_index + 1] = True  # Probability trace visible

        # --- Add Button to List ---
        buttons.append(dict(
            label=t,
            method='update',
            args=[
                {'visible': visible_list},
                # Dynamic title update
                {'title': f"{t}: Price vs {prob_short_name}"} 
            ]
        ))

        first = False
        num += 1

    # --- Layout & Dropdown Setup ---
    fig.update_layout(
        title="Select a Ticker",
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.15, xanchor='left')],
        # Define the two y-axes
        yaxis=dict(title="Price", showgrid=False),
        yaxis2=dict(
            title=probability_column.replace('_', ' '), # Title: e.g., "Buy Probability"
            overlaying='y', 
            side='right', 
            range=[0,1],
            showgrid=True 
        ),
        hovermode='x unified',        
        hoverdistance=100,           
        spikedistance=100             
    )

    # --- X-Axis Spike Setup ---
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',   
        spikesnap='cursor',   
        spikethickness=1,
        spikecolor="grey",
        showgrid=True
    )
    
    return fig
