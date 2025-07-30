import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
from tqdm import tqdm  # Using the standard tqdm for sequential loops
from openai import AsyncOpenAI
import os
import sys
import random

# Add parent directory to path to allow imports from agent_gold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_gold.data_collector.get_sentiment_history import get_headlines_for_past_days
from agent_gold.data_collector.utils import get_sentiment_async

async def get_sentiment_with_retry(headline, client, max_retries=5, initial_delay=1.0):
    """
    Calls get_sentiment_async with exponential backoff and jitter.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(delay * 0.1)
            return await get_sentiment_async(headline, client)
        except Exception as e:
            if "Request timed out" in str(e) and attempt < max_retries - 1:
                print(f"Warning: Request timed out for '{headline[:50]}...'. Retrying (attempt {attempt + 1}/{max_retries})...")
                delay *= 2
                await asyncio.sleep(delay + random.uniform(0, 1))
            else:
                print(f"Error: Could not analyze headline '{headline[:50]}...' after {max_retries} attempts. Error: {e}")
                return None
    return None

async def run_24h_sentiment_sequential(query, num_runs=20, interval_seconds=5):
    """
    Runs sentiment analysis sequentially with an interval to avoid rate-limiting.
    """
    print("--- Sequential 24-Hour Sentiment Analysis ---")
    print(f"Running {num_runs} iterations with a {interval_seconds}-second interval...")

    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    all_sentiments = []

    for i in tqdm(range(num_runs), desc="Sequential Sentiment Runs"):
        # 1. Fetch headlines
        headlines_with_ts = await get_headlines_for_past_days(query, days=2)
        now = datetime.now(timezone.utc)
        past_24h_limit = now - timedelta(hours=24)
        
        filtered_headlines = [
            h['title'] for h in headlines_with_ts 
            if h['timestamp'] > past_24h_limit and h['title']
        ]

        if not filtered_headlines:
            all_sentiments.append(0)
            continue

        # 2. Analyze sentiment for the filtered headlines
        tasks = [get_sentiment_with_retry(h, async_client) for h in filtered_headlines]
        sentiments = await asyncio.gather(*tasks)
        sentiments = [s for s in sentiments if s is not None]

        if not sentiments:
            all_sentiments.append(0)
            continue

        bullish_count = sentiments.count("Bullish")
        bearish_count = sentiments.count("Bearish")
        total_articles = len(sentiments)
        
        run_sentiment = (bullish_count - bearish_count) / total_articles if total_articles > 0 else 0
        all_sentiments.append(run_sentiment)

        # 3. Wait for the specified interval before the next run
        if i < num_runs - 1:
            await asyncio.sleep(interval_seconds)

    # Calculate statistics
    mean_sentiment = np.mean(all_sentiments)
    std_dev_sentiment = np.std(all_sentiments)
    
    print("\n--- Aggregated 24-Hour Sentiment Analysis ---")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of Runs: {num_runs}")
    print(f"Mean Sentiment Score: {mean_sentiment:.4f}")
    print(f"Standard Deviation: {std_dev_sentiment:.4f}")
    
    results_df = pd.DataFrame({
        'timestamp': [datetime.now().isoformat()],
        'mean_sentiment': [mean_sentiment],
        'std_dev_sentiment': [std_dev_sentiment],
        'num_runs': [num_runs]
    })
    
    output_filename = "sentiment_past_24h_sequential.json"
    results_df.to_json(output_filename, orient='records', indent=4)
    print(f"\nResults saved to {output_filename}")

if __name__ == "__main__":
    SEARCH_QUERY = "gold price"
    NUM_RUNS = 20
    INTERVAL_SECONDS = 5
    
    asyncio.run(run_24h_sentiment_sequential(SEARCH_QUERY, NUM_RUNS, INTERVAL_SECONDS))
