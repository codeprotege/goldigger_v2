import pandas as pd
import json

def load_data(price_file="price_data.json", sentiment_file="sentiment_data.json"):
    """
    Loads the price and sentiment data from their respective JSON files.
    """
    try:
        with open(price_file, 'r') as f:
            price_data = json.load(f)
        
        columns = pd.MultiIndex.from_tuples(price_data['columns'])
        price_df = pd.DataFrame(price_data['data'], columns=columns)
        price_df.index = pd.to_datetime(price_data['index'], unit='ms').date
        price_df.columns = price_df.columns.get_level_values(0)
        price_df.index.name = "Date"
        print("Price data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {price_file} not found.")
        return None, None
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None, None

    try:
        sentiment_df = pd.read_json(sentiment_file)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
        print("Sentiment data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {sentiment_file} not found.")
        return None, None
    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        return None, None
        
    return price_df, sentiment_df

def prepare_final_dataset(price_df, sentiment_df, output_filename="final_data.json"):
    """
    Performs feature engineering and merges the datasets.
    """
    # 1. Calculate Price_Direction
    price_df['Price_Direction'] = (price_df['Close'] > price_df['Open']).astype(int)
    
    # 2. Prepare for merging
    # We need the sentiment from day T and the price direction from day T+1
    # So, we shift the price data by one day.
    price_target_df = price_df[['Price_Direction']].copy()
    price_target_df.index = price_target_df.index - pd.to_timedelta(1, unit='d')
    
    # 3. Merge the datasets
    # We merge the sentiment data with the shifted price direction data.
    final_df = pd.merge(sentiment_df, price_target_df, on="Date", how="inner")
    
    # 4. Save the final dataset
    final_df.to_json(output_filename, orient="records", indent=4)
    print(f"\nFinal dataset saved to {output_filename}")
    
    return final_df

if __name__ == "__main__":
    price_df, sentiment_df = load_data()
    
    if price_df is not None and sentiment_df is not None:
        print("\n--- Price Data ---")
        print(price_df.head())
        
        print("\n--- Sentiment Data ---")
        print(sentiment_df.head())
        
        final_dataset = prepare_final_dataset(price_df, sentiment_df)
        
        print("\n--- Final Merged Dataset ---")
        print(final_dataset.head())