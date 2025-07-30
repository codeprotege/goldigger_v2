import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

def load_final_data(filename="final_data.json"):
    """
    Loads the final, merged dataset.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print("Final data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading final data: {e}")
        return None

def train_and_evaluate_model(df):
    """
    Trains a logistic regression model and evaluates its accuracy.
    """
    if df is None or df.empty:
        print("Cannot train model on empty or invalid data.")
        return

    # Define features (X) and target (y)
    features = ['Sentiment_Score', 'Bullish_Count', 'Bearish_Count', 'Neutral_Count', 'Total_Articles']
    target = 'Price_Direction'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    final_df = load_final_data("data_collector/final_data.json")
    
    if final_df is not None:
        train_and_evaluate_model(final_df)
