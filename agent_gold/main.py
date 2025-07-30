from pocketflow import Node, BatchNode, Flow
from utils import get_sentiment
from googlesearch import search
import requests
from bs4 import BeautifulSoup

class FetchNewsNode(Node):
    def exec(self, shared):
        print("Fetching live financial news URLs for Gold for today (2025-07-23)...")
        # Using Google search dorks to filter by date
        query = "latest market news gold price after:2025-07-22 before:2025-07-24"
        try:
            # Fetch live URLs from Google search
            urls = [result for result in search(query, num_results=10, lang="en")]
            if not urls:
                print("No URLs found from live search for today's date.")
                return []
            
            print(f"Found {len(urls)} live URLs.")
            return urls
        except Exception as e:
            print(f"An error occurred while fetching live news: {e}")
            print("Please ensure you have a working internet connection and that the googlesearch library is not being blocked.")
            return []

class ScrapeHeadlineNode(BatchNode):
    def exec(self, url):
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the title tag, which usually contains the main headline
            title = soup.find('title')
            if title:
                return title.get_text(strip=True)
            else:
                return f"No title found for {url}"
        except requests.RequestException as e:
            return f"Could not fetch URL: {url} ({e})"
        except Exception as e:
            return f"An error occurred while scraping {url}: {e}"

class AnalyzeSentimentNode(BatchNode):
    def exec(self, headline):
        # This node now processes each headline scraped by ScrapeHeadlineNode
        if not headline or "Could not fetch" in headline or "No title found" in headline:
            return {"headline": headline, "sentiment": "Error"}
            
        print(f"Analyzing: {headline}")
        sentiment = get_sentiment(headline)
        return {"headline": headline, "sentiment": sentiment}

    def post(self, shared, prep_res, exec_res):
        return "default"

class DisplayResultsNode(Node):
    def exec(self, results):
        print("\n--- Market Sentiment Analysis ---")
        
        valid_results = [r for r in results if r and r['sentiment'] != 'Error']
        
        bullish_count = sum(1 for r in valid_results if r['sentiment'] == 'Bullish')
        bearish_count = sum(1 for r in valid_results if r['sentiment'] == 'Bearish')
        neutral_count = sum(1 for r in valid_results if r['sentiment'] == 'Neutral')
        error_count = len(results) - len(valid_results)

        for result in results:
            print(f"[{result['sentiment']}] {result['headline']}")
        
        print("\n--- Summary ---")
        print(f"Bullish: {bullish_count}")
        print(f"Bearish: {bearish_count}")
        print(f"Neutral: {neutral_count}")
        print(f"Errors: {error_count}")
        print("-----------------")
        return "done"

# --- Create and Run the Flow ---
# 1. Instantiate the nodes
fetch_news = FetchNewsNode()
scrape_headlines = ScrapeHeadlineNode()
analyze_sentiment = AnalyzeSentimentNode()
display_results = DisplayResultsNode()

# 2. Define the new workflow
fetch_news >> scrape_headlines >> analyze_sentiment >> display_results

# 3. Create the flow, starting with the first node
flow = Flow(start=fetch_news)

# 4. Run the agent
if __name__ == "__main__":
    print("Starting Market Sentiment Agent...")
    shared_data = {}
    flow.run(shared_data)
    print("\nAgent run complete.")
