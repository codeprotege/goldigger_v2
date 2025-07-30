# Project Blueprint: Sentiment-Based Gold Price Prediction

This document outlines the plan and progress for building a machine learning model that predicts the price direction of gold based on the sentiment of financial news.

## Objective

To train a machine learning model that ingests the aggregated sentiment from one day's news to predict whether the price of gold will go up or down on the following day.

## Project Plan

The project is divided into three main stages:

### Stage 1: Historical Data Collection

This is the foundational phase, focused on gathering two parallel datasets: historical news sentiment and historical market prices.

1.  **Sentiment Data Collection:**
    *   **Method:** Create a dedicated data collection script.
    *   **Process:**
        *   Loop through each day in a specified historical range (e.g., the last year).
        *   For each day, construct a Google search query to find news published only on that day.
        *   Fetch the resulting URLs.
        *   Scrape the headline from each URL.
        *   Analyze the sentiment of each headline ("Bullish," "Bearish," "Neutral").
        *   Calculate and store a daily aggregate sentiment score.
        *   Save the daily data (`Date`, `Sentiment_Score`) to a `sentiment_data.csv` file.

2.  **Price Data Collection:**
    *   **Method:** Use the `yfinance` Python library.
    *   **Process:**
        *   Fetch the daily Open, High, Low, and Close prices for Gold (ticker: `GC=F`) for the same historical range.
        *   Save this data (`Date`, `Open`, `Close`, etc.) to a `price_data.csv` file.

---

### Stage 2: Feature Engineering & Data Alignment

This stage involves cleaning, transforming, and merging the datasets to prepare them for the machine learning model.

1.  **Calculate Sentiment Score:** For each day in `sentiment_data.csv`, create a single numerical score. A good starting point is the **Net Sentiment Score**:
    *   `Sentiment Score = (Number of Bullish - Number of Bearish) / Total Articles`
    *   This creates a value between -1 (very bearish) and +1 (very bullish).

2.  **Define the Prediction Target:** In `price_data.csv`, create the target variable, `Price_Direction`.
    *   If `Close Price > Open Price`, `Price_Direction` = `1` (Up).
    *   If `Close Price < Open Price`, `Price_Direction` = `0` (Down).

3.  **Merge Datasets:**
    *   Combine the two datasets into a single master CSV file, aligned by date.
    *   Crucially, the `Sentiment_Score` from Day `T` will be matched with the `Price_Direction` from Day `T+1`. This is because we are using yesterday's news to predict today's price movement.

---

### Stage 3: Machine Learning Model

With the final, prepared dataset, we will train and evaluate a classification model.

1.  **Choose a Model:** Start with a standard, interpretable model from `scikit-learn`, such as `LogisticRegression` or `RandomForestClassifier`.
2.  **Train/Test Split:** Divide the data chronologically. Use the older 80% for training the model and the most recent 20% for testing its performance on unseen data.
3.  **Evaluation:** Assess the model's accuracy. How often did it correctly predict whether the price would go up or down?

## Progress Log

*   **2025-07-23:**
    *   Successfully built and debugged `agent_gold`, a PocketFlow agent capable of fetching live news URLs for a specific topic (gold) and date, scraping the headlines, and analyzing their sentiment.
    *   **Decision:** The project will now evolve from a single-day analysis tool into a historical data collection and prediction pipeline. The plan above has been formulated.
    *   Created the `data_collector` directory to house the new scripts.
    *   Created `get_price_history.py` to download historical gold prices from Yahoo Finance.
    *   Created `get_sentiment_history.py` to scrape historical news headlines using Selenium and BeautifulSoup.
    *   Updated `requirements.txt` with all necessary libraries for data collection.

---
Next Step: Execute the data collection scripts, starting with `get_price_history.py`.
