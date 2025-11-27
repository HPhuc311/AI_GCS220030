import joblib
import pandas as pd
import json
import os
from typing import List
import datetime

MODEL_PATH = "titanic_model.joblib"
DATA_PATH = "titanic.csv" 

pipeline = None
df = None

# Attempt to load the model and data upon script initialization
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        pipeline = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"WARNING: Could not find {MODEL_PATH} or {DATA_PATH}. Analysis/Prediction features will not work.")
except Exception as e:
    print(f"Error loading resources: {e}")

def get_data_statistics(column_name: str) -> str:
    """
    Provides basic statistics (mean, min, max, count) or the top 5 frequency values for a specific column.

    Args:
        column_name: The name of the column to analyze (e.g., 'Age', 'Fare', 'Sex').

    Returns:
        A JSON string containing the statistics or frequency.
    """
    if df is None:
        return json.dumps({"error": "Data has not been loaded. Please check the 'titanic.csv' file."})
        
    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found in the dataset. Available columns: {list(df.columns)}"})

    col = df[column_name]

    if pd.api.types.is_numeric_dtype(col):
        # Numeric column: calculate basic statistics
        stats = {
            "column": column_name,
            "count": int(col.count()),
            "mean": round(col.mean(), 2) if col.count() > 0 else None,
            "min": round(col.min(), 2) if col.count() > 0 else None,
            "max": round(col.max(), 2) if col.count() > 0 else None
        }
    else:
        # Categorical column: calculate top 5 value frequencies
        value_counts = col.value_counts().head(5)
        stats = {
            "column": column_name,
            "count": int(col.count()),
            "top_frequencies": value_counts.apply(int).to_dict() # Convert to standard int for JSON
        }
        
    return json.dumps(stats, indent=4)

def predict_survival(Pclass: int, Sex: str, Age: float, Fare: float, Embarked: str) -> str:
    """
    Predicts the survival probability (0=Perished, 1=Survived) on the Titanic based on the input features.

    Args:
        Pclass: Passenger Class (1, 2, or 3).
        Sex: Gender ('male' or 'female').
        Age: Age of the passenger (float).
        Fare: Fare price (float).
        Embarked: Port of embarkation ('S', 'C', or 'Q').

    Returns:
        A JSON string containing the prediction result and probabilities.
    """
    if pipeline is None:
        return json.dumps({"error": "Prediction model not loaded. Please run ml_model.py."})

    # Create a DataFrame from the input parameters
    input_data = pd.DataFrame([{
        'Pclass': Pclass,
        'Sex': Sex.lower(),
        'Age': Age,
        'Fare': Fare,
        'Embarked': Embarked.upper()
    }])

    try:
        # Prediction
        prediction = pipeline.predict(input_data)[0]
        # Get the probability for each class (0 and 1)
        probability = pipeline.predict_proba(input_data)[0].tolist() 

        result = {
            "input": input_data.iloc[0].to_dict(),
            "prediction_code": int(prediction),
            "prediction_label": "Survived" if prediction == 1 else "Perished",
            "probability_perished": round(probability[0], 4),
            "probability_survived": round(probability[1], 4)
        }
        return json.dumps(result, indent=4)
        
    except Exception as e:
        return json.dumps({"error": f"Prediction failed due to an internal error: {e}"})

def get_current_datetime() -> str:
    """
    Returns the current date and time in a human-readable format, 
    including the day, month, year, and time.
    (Returns JSON string)
    """
    now = datetime.datetime.now()
    result = {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": now.timestamp(),
        "timezone": "Local"
    }
    return json.dumps(result, indent=4)

def get_weather_forecast(city: str) -> str:
    """
    Provides the weather forecast for a specified city. 
    (Note: This is a simulated function and returns static data.)
    
    Args:
        city: The name of the city (e.g., 'Hanoi', 'London').
        
    Returns:
        A JSON string containing the simulated forecast.
    """
    # Static simulation for demonstration purposes
    simulated_data = {
        "city": city,
        "temperature": "25°C",
        "condition": "Mostly sunny with light breeze.",
        "humidity": "65%",
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return json.dumps(simulated_data, indent=4)

# List of all available functions for the Gemini model
ALL_TOOLS: List = [get_data_statistics, predict_survival, get_current_datetime, get_weather_forecast ]