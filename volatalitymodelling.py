import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# List of stock tickers
tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "HCLTECH.NS", "BAJFINANCE.NS", "ONGC.NS", "AXISBANK.NS", 
    "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "KOTAKBANK.NS", "NTPC.NS"
]

# Define the historical date range
start_date = "2005-01-01"
end_date = "2018-07-31"

# Download historical data
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Extract the 'Adj Close' prices
adj_close_data = {ticker: data[ticker]['Adj Close'] for ticker in tickers}

# Convert to DataFrame
df = pd.DataFrame(adj_close_data)

# Calculate daily returns
daily_returns = df.pct_change()

# Resample to weekly returns and calculate weekly volatility
weekly_volatility = daily_returns.resample('W').std()

# Define the forecast period
forecast_start_date = "2018-09-01"
forecast_end_date = "2024-02-29"

# Download actual data for the forecast period
actual_data = yf.download(tickers, start=forecast_start_date, end=forecast_end_date, group_by='ticker')

# Extract the 'Adj Close' prices for the actual data
actual_adj_close_data = {ticker: actual_data[ticker]['Adj Close'] for ticker in tickers}

# Convert to DataFrame
actual_df = pd.DataFrame(actual_adj_close_data)

# Calculate actual daily returns and weekly volatility for the forecast period
actual_daily_returns = actual_df.pct_change()
actual_weekly_volatility = actual_daily_returns.resample('W').std()

# Combine historical and actual data
combined_volatility = pd.concat([weekly_volatility, actual_weekly_volatility])

# Function to forecast volatility using GARCH in a rolling window manner
def rolling_forecast_volatility(volatility_series, forecast_start, forecast_end, actual_volatility_series, ticker):
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='W')
    forecasts = []
    
    for date in forecast_dates:
        # Select data up to the current date
        current_data = volatility_series.loc[:date].dropna()
        
        # Rescale the data
        rescaled_data = 100 * current_data
        
        # Fit GARCH model
        garch_model = arch_model(rescaled_data, vol='Garch', p=1, q=1)
        model_fit = garch_model.fit(disp='off')
        
        # Forecast the next week's volatility
        forecast = model_fit.forecast(horizon=1)
        forecast_volatility = np.sqrt(forecast.variance.values[-1, 0]) / 100  # Rescale back
        
        forecasts.append(forecast_volatility)
    
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Volatility': forecasts})
    
    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(volatility_series.index, volatility_series, label='Historical Volatility', color='blue')
    plt.plot(forecast_df['Date'], forecast_df['Forecasted Volatility'], label='Forecasted Volatility', color='red')
    plt.plot(actual_volatility_series.index, actual_volatility_series, label='Actual Volatility', color='gray', linestyle='--')
    plt.title(f'Weekly Volatility Forecast for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend(loc='upper left')
    plt.show()
    
    return forecast_df

# Forecast, plot, and compare volatility for each stock
forecasted_volatility_rolling = {}

for ticker in tickers:
    print(f"Rolling forecast for {ticker}")
    volatility_series = combined_volatility[ticker]
    actual_volatility_series = actual_weekly_volatility[ticker]
    forecasted_volatility_rolling[ticker] = rolling_forecast_volatility(volatility_series, forecast_start_date, forecast_end_date, actual_volatility_series, ticker)
