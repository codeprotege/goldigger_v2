#!/usr/bin/env python3
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from openai import AsyncOpenAI
import os
import sys
import random
import requests
import json
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Import search functionality
try:
    from pygoogle import search as pygoogle_search
    SEARCH_METHOD = 'pygoogle'
    print("🔍 Using PyGoogle for search")
except ImportError:
    try:
        from googlesearch import search
        SEARCH_METHOD = 'googlesearch'
        print("🔍 Using googlesearch-python")
    except ImportError:
        print("❌ No Google search library available")
        sys.exit(1)

def extract_article_content(url, timeout=15):
    """Extract full article content from URL for deeper analysis"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find article content
        content = ""
        
        # Method 1: Look for article tags
        article = soup.find('article')
        if article:
            content = article.get_text(strip=True)
        
        # Method 2: Look for main content div
        elif soup.find('div', class_=['content', 'article-content', 'post-content', 'story-content']):
            content_div = soup.find('div', class_=['content', 'article-content', 'post-content', 'story-content'])
            content = content_div.get_text(strip=True)
        
        # Method 3: Look for paragraphs
        elif soup.find_all('p'):
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])  # First 5 paragraphs
        
        # Clean and limit content
        if content:
            content = ' '.join(content.split())  # Remove extra whitespace
            content = content[:2000]  # Limit to 2000 characters
            return content
        else:
            return None
            
    except Exception as e:
        return None

async def balanced_sentiment_analysis(headline, client, article_content=None, date_str=None):
    """
    使用平衡方法分析情緒，減少偏見
    """
    if not headline:
        return {'sentiment': 'Neutral', 'confidence': 'Low', 'factors': 'No headline provided', 'raw_response': 'Error'}
    
    # 首先使用關鍵詞方法
    headline_lower = headline.lower()
    
    # 強烈看跌關鍵詞
    bearish_strong = ['falls', 'drops', 'dips', 'declines', 'tumbles', 'plunges', 'slides', 'slumps', 'crashes', 'collapses']
    # 強烈看漲關鍵詞  
    bullish_strong = ['rises', 'gains', 'climbs', 'surges', 'jumps', 'rallies', 'soars', 'record high', 'all-time high', 'breaks higher']
    # 中等看跌
    bearish_mod = ['lower', 'down', 'weak', 'pressure', 'selling', 'drags', 'pin', 'below', 'ebb', 'retreats']
    # 中等看漲
    bullish_mod = ['higher', 'up', 'strong', 'support', 'buying', 'firms', 'steadies', 'holds', 'supported']
    # 中性
    neutral_words = ['holds steady', 'unchanged', 'flat', 'mixed', 'consolidate', 'sideways', 'range-bound']
    
    # 關鍵詞分析
    if any(word in headline_lower for word in bearish_strong):
        keyword_sentiment = 'Bearish'
        keyword_confidence = 'High'
    elif any(word in headline_lower for word in bullish_strong):
        keyword_sentiment = 'Bullish'
        keyword_confidence = 'High'
    elif any(word in headline_lower for word in bearish_mod):
        keyword_sentiment = 'Bearish'
        keyword_confidence = 'Medium'
    elif any(word in headline_lower for word in bullish_mod):
        keyword_sentiment = 'Bullish'
        keyword_confidence = 'Medium'
    elif any(word in headline_lower for word in neutral_words):
        keyword_sentiment = 'Neutral'
        keyword_confidence = 'High'
    else:
        keyword_sentiment = 'Neutral'
        keyword_confidence = 'Low'
    
    # 如果關鍵詞分析信心度高，直接使用
    if keyword_confidence == 'High':
        return {
            'sentiment': keyword_sentiment,
            'confidence': keyword_confidence,
            'factors': f'Direct {keyword_sentiment.lower()} language in headline',
            'raw_response': 'Keyword analysis'
        }
    
    # 否則使用AI分析作為補充
    content_text = f"\n\nArticle Content: {article_content[:1000]}..." if article_content else ""
    date_context = f"\n\nDate: {date_str}" if date_str else ""
        
    prompt = f"""
    You are an objective financial analyst. Analyze this gold/XAUUSD news headline for market sentiment.

    CRITICAL INSTRUCTIONS:
    1. Focus PRIMARILY on what the headline directly states about gold price movement
    2. If headline says gold "falls", "drops", "dips", "declines" = BEARISH
    3. If headline says gold "rises", "gains", "climbs", "surges" = BULLISH
    4. If headline is neutral or mixed = NEUTRAL
    5. Consider broader economic context as SECONDARY factor only

    HEADLINE: "{headline}"{content_text}{date_context}

    Respond ONLY with: [Sentiment]|[Confidence]|[Brief reasoning]
    Where Sentiment is: Bullish, Bearish, or Neutral
    Where Confidence is: High, Medium, or Low
    """
    
    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        
        if '|' in result:
            parts = result.split('|')
            if len(parts) >= 3:
                ai_sentiment = parts[0].strip()
                ai_confidence = parts[1].strip()
                factors = parts[2].strip()
                
                # 如果AI和關鍵詞分析一致，提高信心度
                if ai_sentiment == keyword_sentiment:
                    final_confidence = 'High' if ai_confidence in ['High', 'Medium'] else 'Medium'
                else:
                    # 如果不一致，使用關鍵詞結果但降低信心度
                    final_confidence = 'Medium' if keyword_confidence == 'High' else 'Low'
                
                return {
                    'sentiment': keyword_sentiment,  # 優先使用關鍵詞結果
                    'confidence': final_confidence,
                    'factors': factors,
                    'raw_response': result
                }
        
        # 回退到關鍵詞分析
        return {
            'sentiment': keyword_sentiment,
            'confidence': keyword_confidence,
            'factors': f'Keyword-based {keyword_sentiment.lower()} analysis',
            'raw_response': 'AI analysis failed, used keyword fallback'
        }
            
    except Exception as e:
        return {
            'sentiment': keyword_sentiment,
            'confidence': 'Low',
            'factors': f'Analysis error, used keyword fallback: {keyword_sentiment.lower()}',
            'raw_response': str(e)
        }

def scrape_headline_from_url(url, timeout=10):
    """Scrape headline/title from a given URL with improved error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try multiple methods to find the headline
        title = None
        
        # Method 1: title tag
        if soup.find('title'):
            title = soup.find('title').get_text(strip=True)
        
        # Method 2: h1 tag
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        
        # Method 3: meta og:title
        elif soup.find('meta', property='og:title'):
            title = soup.find('meta', property='og:title').get('content', '').strip()
        
        # Method 4: meta name="title"
        elif soup.find('meta', attrs={'name': 'title'}):
            title = soup.find('meta', attrs={'name': 'title'}).get('content', '').strip()
        
        # Method 5: article title
        elif soup.find('h1', class_=['title', 'headline', 'article-title']):
            title = soup.find('h1', class_=['title', 'headline', 'article-title']).get_text(strip=True)
        
        if title and len(title) > 10:
            # Clean up title
            title = title.replace('\n', ' ').replace('\r', ' ')
            title = ' '.join(title.split())  # Remove extra spaces
            return title
        else:
            return None
            
    except Exception as e:
        return None

def search_gold_news_today(max_results=10):
    """Search for today's gold/XAUUSD news"""
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')
    date_chinese = today.strftime('%Y年%m月%d日')
    
    # Search queries for today's gold news
    search_queries = [
        f"XAUUSD gold price today {date_str} news analysis",
        f"gold market today Fed Powell inflation July 2025",
        f"黃金價格 今日 {date_chinese} 分析 美元 走勢",
        f"gold trading today geopolitical tensions safe haven",
        f"XAUUSD technical analysis today resistance support 2025",
        f"precious metals today central bank policy rates",
        f"gold outlook today Treasury yields interest rates July"
    ]
    
    all_urls = []
    headlines_data = []
    
    print(f"🔍 Using {SEARCH_METHOD} to search for today's gold news ({date_str})...")
    
    for i, search_query in enumerate(search_queries):
        try:
            print(f"  [{i+1}/{len(search_queries)}] {search_query[:55]}...")
            
            if SEARCH_METHOD == 'pygoogle':
                search_results = pygoogle_search(search_query, num_results=3)
                urls = [result.link for result in search_results if hasattr(result, 'link')]
            else:
                urls = list(search(search_query, num_results=3, lang="en"))
            
            if urls:
                all_urls.extend(urls)
                print(f"    Found {len(urls)} URLs")
                time.sleep(random.uniform(1.2, 2.0))  # Respectful delay
            else:
                print(f"    No URLs found")
            
        except Exception as e:
            print(f"    Search error: {e}")
            continue
    
    # Remove duplicates
    unique_urls = list(set(all_urls))
    print(f"  Total unique URLs: {len(unique_urls)}")
    
    # Scrape headlines and content
    print(f"\n📰 Scraping headlines and content...")
    for i, url in enumerate(unique_urls[:max_results]):
        try:
            print(f"  [{i+1}/{min(len(unique_urls), max_results)}] {urlparse(url).netloc}")
            headline = scrape_headline_from_url(url)
            
            if headline:
                headline_lower = headline.lower()
                
                # Gold-related keywords
                gold_keywords = ['gold', 'xau', 'precious', 'metal', 'bullion', 'commodity', '黃金', '貴金屬', '金價']
                relevance_keywords = ['price', 'trading', 'market', 'analysis', 'forecast', 'outlook', '價格', '交易', '市場', '分析', '預測']
                
                has_gold = any(keyword in headline_lower for keyword in gold_keywords)
                has_relevance = any(keyword in headline_lower for keyword in relevance_keywords)
                
                # Priority scoring
                sentiment_keywords = ['gains', 'rises', 'surge', 'rally', 'falls', 'drops', 'dip', 'decline', 
                                    'bullish', 'bearish', 'breakthrough', 'resistance', 'support', 
                                    '上漲', '下跌', '看漲', '看跌', '突破', '支撐', '阻力']
                has_sentiment = any(keyword in headline_lower for keyword in sentiment_keywords)
                
                if has_gold or has_relevance:
                    print(f"    📄 Extracting content...")
                    article_content = extract_article_content(url)
                    content_status = "✅ Content extracted" if article_content else "❌ No content"
                    
                    priority = 3 if (has_gold and has_sentiment) else 2 if has_gold else 1
                    
                    headlines_data.append({
                        'title': headline,
                        'source': urlparse(url).netloc,
                        'url': url,
                        'priority': priority,
                        'has_sentiment_keywords': has_sentiment,
                        'article_content': article_content
                    })
                    
                    priority_stars = "⭐" * priority
                    print(f"    {priority_stars} {headline}")
                    print(f"    {content_status}")
                else:
                    print(f"    ❌ Not relevant: {headline}")
            else:
                print(f"    ❌ Could not extract headline")
            
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Sort by priority
    headlines_data.sort(key=lambda x: x['priority'], reverse=True)
    
    return headlines_data

async def analyze_todays_gold_news():
    """Analyze today's gold news with balanced sentiment analysis"""
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')
    day_name = today.strftime('%A')
    
    print("🚀 Today's Gold Sentiment Analysis")
    print("=" * 50)
    print(f"📅 Date: {day_name}, {date_str}")
    print(f"🎯 Method: Balanced Sentiment Analysis (Anti-Bias)")
    
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in .env file")
        return
    
    # Initialize client
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Search for today's news
    headlines_data = search_gold_news_today(max_results=10)
    
    if not headlines_data:
        print(f"❌ No relevant headlines found for today")
        return
    
    headlines = [h['title'] for h in headlines_data]
    print(f"\n📊 Found {len(headlines)} relevant headlines for analysis")
    
    # Analyze sentiment
    print(f"\n🤖 Performing balanced sentiment analysis...")
    tasks = []
    for headline_data in headlines_data:
        task = balanced_sentiment_analysis(
            headline_data['title'], 
            client, 
            headline_data.get('article_content'),
            date_str
        )
        tasks.append(task)
    
    sentiment_results = await asyncio.gather(*tasks)
    
    # Process results
    detailed_sentiments = []
    bullish_count = bearish_count = neutral_count = 0
    has_content_count = high_confidence_count = 0
    
    for headline_data, sentiment_result in zip(headlines_data, sentiment_results):
        sentiment = sentiment_result.get('sentiment', 'Neutral')
        confidence = sentiment_result.get('confidence', 'Medium')
        factors = sentiment_result.get('factors', 'Analysis completed')
        
        if headline_data.get('article_content'):
            has_content_count += 1
        if confidence == 'High':
            high_confidence_count += 1
        
        detailed_sentiments.append({
            'headline': headline_data['title'],
            'sentiment': sentiment,
            'confidence': confidence,
            'factors': factors,
            'source': headline_data['source'],
            'priority': headline_data['priority'],
            'has_content': bool(headline_data.get('article_content'))
        })
        
        if sentiment == "Bullish":
            bullish_count += 1
        elif sentiment == "Bearish":
            bearish_count += 1
        else:
            neutral_count += 1
    
    total_articles = len(sentiment_results)
    sentiment_score = (bullish_count - bearish_count) / total_articles if total_articles > 0 else 0
    
    # Display results
    print(f"\n📈 Today's Results ({date_str}):")
    print(f"   Total Headlines: {total_articles}")
    print(f"   Bullish: {bullish_count} ({bullish_count/total_articles*100:.1f}%)")
    print(f"   Bearish: {bearish_count} ({bearish_count/total_articles*100:.1f}%)")
    print(f"   Neutral: {neutral_count} ({neutral_count/total_articles*100:.1f}%)")
    print(f"   Sentiment Score: {sentiment_score:.4f}")
    print(f"   Content Extracted: {has_content_count}/{total_articles}")
    print(f"   High Confidence: {high_confidence_count}/{total_articles}")
    
    # Sentiment interpretation
    if sentiment_score > 0.5:
        sentiment_label = 'Strong Bullish'
        emoji = '📈📈'
    elif sentiment_score > 0.2:
        sentiment_label = 'Moderate Bullish'
        emoji = '📈'
    elif sentiment_score > -0.2:
        sentiment_label = 'Neutral'
        emoji = '➡️'
    elif sentiment_score > -0.5:
        sentiment_label = 'Moderate Bearish'
        emoji = '📉'
    else:
        sentiment_label = 'Strong Bearish'
        emoji = '📉📉'
    
    print(f"\n{emoji} Overall Sentiment: {sentiment_label}")
    
    # Show detailed analysis
    print(f"\n📰 Detailed Analysis:")
    for i, detail in enumerate(detailed_sentiments, 1):
        confidence_emoji = "🎯" if detail['confidence'] == 'High' else "🔍" if detail['confidence'] == 'Medium' else "❓"
        sentiment_emoji = "📈" if detail['sentiment'] == 'Bullish' else "📉" if detail['sentiment'] == 'Bearish' else "➡️"
        
        print(f"\n{i}. {sentiment_emoji} {detail['sentiment']} {confidence_emoji} ({detail['confidence']})")
        print(f"   📰 {detail['headline']}")
        print(f"   🌐 {detail['source']}")
        print(f"   💭 {detail['factors']}")
    
    # Save results
    today_results = {
        'analysis_date': datetime.now().isoformat(),
        'target_date': date_str,
        'day_of_week': day_name,
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'total_headlines': total_articles,
        'headlines': headlines,
        'sentiment_details': detailed_sentiments,
        'has_content_count': has_content_count,
        'high_confidence_count': high_confidence_count,
        'method': 'Balanced Anti-Bias Analysis'
    }
    
    # Save JSON
    json_filename = f"todays_gold_sentiment_{date_str}.json"
    with open(json_filename, 'w') as f:
        json.dump(today_results, f, indent=2, default=str)
    
    # Save CSV
    csv_data = [{
        'Date': date_str,
        'Day_of_Week': day_name,
        'Sentiment_Score': sentiment_score,
        'Sentiment_Label': sentiment_label,
        'Total_Headlines': total_articles,
        'Bullish_Count': bullish_count,
        'Bearish_Count': bearish_count,
        'Neutral_Count': neutral_count,
        'Bullish_Percentage': round(bullish_count/total_articles*100, 1),
        'Bearish_Percentage': round(bearish_count/total_articles*100, 1),
        'Neutral_Percentage': round(neutral_count/total_articles*100, 1),
        'Content_Extracted': has_content_count,
        'High_Confidence': high_confidence_count,
        'Top_Headlines': '; '.join(headlines[:3])
    }]
    
    csv_df = pd.DataFrame(csv_data)
    csv_filename = f"todays_gold_sentiment_{date_str}.csv"
    csv_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Results saved:")
    print(f"   📄 JSON: {json_filename}")
    print(f"   📊 CSV: {csv_filename}")
    
    print(f"\n✨ Today's analysis complete!")
    return today_results

if __name__ == "__main__":
    asyncio.run(analyze_todays_gold_news())