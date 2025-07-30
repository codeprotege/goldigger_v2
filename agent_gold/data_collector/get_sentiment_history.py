import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pygooglenews import GoogleNews
from tqdm import tqdm
import os
from dotenv import load_dotenv
import sys
from openai import AsyncOpenAI
from email.utils import parsedate_to_datetime

# Load environment variables
load_dotenv()

# Assuming utils.py is in the same directory
from .utils import get_sentiment_async

def get_headlines_for_day(query, date_str):
    """
    Fetches headlines for a specific day by setting a start and end date.
    
    Args:
        query (str): The search query.
        date_str (str): The target date in 'YYYY-MM-DD' format.
    """
    gn = GoogleNews()
    all_headlines = []
    
    # This is a blocking function, so it's best run in an executor in an async context
    try:
        # Define the 24-hour window for the query
        search = gn.search(query, from_=date_str, to_=date_str)
        
        for entry in search['entries']:
            try:
                timestamp = parsedate_to_datetime(entry.published)
                all_headlines.append({'title': entry.title, 'timestamp': timestamp})
            except (TypeError, ValueError):
                continue
    except Exception as e:
        print(f"An error occurred while fetching news for {date_str}: {e}")
        
    return all_headlines

async def get_headlines_for_past_days(query, days=2):
    """
    Asynchronously fetches headlines and their publication timestamps for the last few days.
    """
    gn = GoogleNews()
    loop = asyncio.get_event_loop()
    all_headlines = []

    # This is the blocking function that will be run in a thread
    def search_news():
        try:
            # Fetch news for the specified period
            search = gn.search(query, when=f'{days}d')
            
            # Process entries
            for entry in search['entries']:
                try:
                    # The 'published' field is a string like 'Sat, 26 Jul 2025 12:00:00 GMT'
                    timestamp = parsedate_to_datetime(entry.published)
                    all_headlines.append({'title': entry.title, 'timestamp': timestamp})
                except (TypeError, ValueError):
                    # If parsing fails, skip this article
                    continue
        except Exception as e:
            print(f"An error occurred while fetching news: {e}")
        return all_headlines

    # Run the blocking function in a separate thread
    return await loop.run_in_executor(None, search_news)

async def get_sentiment_for_date(query, single_date, client):
    """
    Collects daily sentiment scores for a given query and a single date.
    Accepts an existing client session.
    """
    # Note: This function now fetches headlines with timestamps but only uses the titles.
    headlines_with_ts = await get_headlines_for_past_days(query, days=1) # Fetch for one day
    headlines = [h['title'] for h in headlines_with_ts if h['timestamp'].date() == single_date.date()]

    if not headlines:
        return {
            "Date": single_date.strftime('%Y-%m-%d'),
            "Sentiment_Score": 0,
            "Bullish_Count": 0,
            "Bearish_Count": 0,
            "Neutral_Count": 0,
            "Total_Articles": 0
        }

    tasks = [get_sentiment_async(h, client) for h in headlines if h]
    sentiments = await asyncio.gather(*tasks)

    bullish_count = sentiments.count("Bullish")
    bearish_count = sentiments.count("Bearish")
    neutral_count = sentiments.count("Neutral")
    
    total_articles = len(headlines)
    net_sentiment = (bullish_count - bearish_count) / total_articles if total_articles > 0 else 0

    return {
        "Date": single_date.strftime('%Y-%m-%d'),
        "Sentiment_Score": net_sentiment,
        "Bullish_Count": bullish_count,
        "Bearish_Count": bearish_count,
        "Neutral_Count": neutral_count,
        "Total_Articles": total_articles
    }

async def main():
    """
    Main function to run sentiment analysis for a specific date and save to a file.
    """
    SEARCH_QUERY = "gold price"
    
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        print("Usage: python -m agent_gold.data_collector.get_sentiment_history [YYYY-MM-DD]")
        sys.exit(1)

    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    daily_data = await get_sentiment_for_date(SEARCH_QUERY, target_date, async_client)
    
    df = pd.DataFrame([daily_data])
    output_filename = f"sentiment_{target_date.strftime('%Y-%m-%d')}.json"
    df.to_json(output_filename, orient="records", indent=4)
    print(f"\nSuccessfully saved sentiment data to {output_filename}")

if __name__ == "__main__":
    asyncio.run(main())

