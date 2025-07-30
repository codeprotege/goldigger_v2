from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Synchronous Version ---
def get_sentiment(headline):
    # Point to the OpenRouter API
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    prompt = f"""
    Analyze the sentiment of the following financial news headline. 
    Classify it as "Bullish", "Bearish", or "Neutral".
    Return only the classification.

    Headline: "{headline}"
    Sentiment:
    """
    
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct", # Using a free model on OpenRouter
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# --- Asynchronous Version ---
async def get_sentiment_async(headline, client):
    """
    Asynchronously analyzes the sentiment of a single headline.
    """
    if not headline:
        return "Neutral" # Or handle as an error
        
    prompt = f"""
    Analyze the sentiment of the following financial news headline. 
    Classify it as "Bullish", "Bearish", or "Neutral".
    Return only the classification.

    Headline: "{headline}"
    Sentiment:
    """
    
    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error analyzing headline '{headline}': {e}")
        return "Neutral" # Fallback sentiment

if __name__ == "__main__":
    import asyncio

    async def main():
        # Test sync version
        test_headline_sync = "Tech stocks surge as inflation fears ease."
        sentiment_sync = get_sentiment(test_headline_sync)
        print(f"--- Sync Test ---")
        print(f"Headline: {test_headline_sync}")
        print(f"Sentiment: {sentiment_sync}\n")

        # Test async version
        print(f"--- Async Test ---")
        async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        test_headline_async = "Gold prices plummet amid strengthening dollar."
        sentiment_async = await get_sentiment_async(test_headline_async, async_client)
        print(f"Headline: {test_headline_async}")
        print(f"Sentiment: {sentiment_async}")

    asyncio.run(main())
