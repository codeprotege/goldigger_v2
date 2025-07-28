import yfinance as yf
import sys
import pandas as pd

def get_current_price(ticker="GC=F"):
    """
    Fetches the most recent price for a given ticker.
    """
    try:
        # Fetch data for the last day, which gives the most recent trading session
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        
        if data.empty:
            print(f"No data found for {ticker} for the most recent trading day.")
            print("This could be because the market is currently closed or it is a holiday.")
            return None

        # Flatten the MultiIndex columns if yfinance returns one
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # The last entry in the dataframe has the most recent price and volume
        latest_price = data['Close'].iloc[-1]
        latest_volume = data['Volume'].iloc[-1]
        timestamp = data.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')
        return latest_price, latest_volume, timestamp
        
    except Exception as e:
        print(f"An error occurred while fetching the price: {e}")
        return None, None, None

if __name__ == "__main__":
    price, volume, ts = get_current_price()
    if price is not None and ts is not None:
        print(f"Gold (GC=F) Latest Price: ${price:.2f}")
        print(f"Volume: {volume:,}")
        print(f"As of: {ts}")

