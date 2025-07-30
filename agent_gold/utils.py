from openai import OpenAI
import os

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

if __name__ == "__main__":
    test_headline = "Tech stocks surge as inflation fears ease."
    sentiment = get_sentiment(test_headline)
    print(f"Headline: {test_headline}")
    print(f"Sentiment: {sentiment}")
