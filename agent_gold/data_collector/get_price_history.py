import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

def get_price_history(ticker, start_date, end_date, filename="price_data.json"):
    """
    Fetches historical market data for a given ticker and saves it to a JSON file.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'GC=F' for Gold).
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        filename (str): The name of the output JSON file.
    """
    print(f"Fetching historical price data for {ticker} from {start_date} to {end_date}...")
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print("No data found for the given ticker and date range.")
            return
            
        # Ensure the index is just the date part
        data.index = data.index.date
        
        # Save to JSON
        data.to_json(filename, orient="split", indent=4)
        print(f"Successfully saved price data to {filename}")
        
    except Exception as e:
        print(f"An error occurred while fetching price data: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    GOLD_TICKER = "GC=F"
    
    # Determine output filename, relative to this script's location
    script_dir = os.path.dirname(__file__)
    output_filename = os.path.join(script_dir, "price_data.json")

    if len(sys.argv) > 1:
        try:
            # Use the provided date
            target_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
            # yfinance end date is exclusive, so add one day to include the target date
            start_str = target_date.strftime('%Y-%m-%d')
            end_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        # Default to fetching data for the past year.
        print("Usage: python get_price_history.py [YYYY-MM-DD]")
        print("Fetching data for the last 365 days by default.")
        END_DATE = datetime.now()
        START_DATE = END_DATE - timedelta(days=365)
        start_str = START_DATE.strftime('%Y-%m-%d')
        end_str = END_DATE.strftime('%Y-%m-%d')

    get_price_history(GOLD_TICKER, start_str, end_str, filename=output_filename)
