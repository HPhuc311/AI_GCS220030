import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

MODEL_PATH = "titanic_model.joblib"
DATA_PATH = "titanic.csv"

def prepare_data():
    """Create or load the Titanic data."""
    if os.path.exists(DATA_PATH):
        print(f"Found file '{DATA_PATH}'. Loading...")
        return pd.read_csv(DATA_PATH)
    else:
        print(f"File '{DATA_PATH}' not found. Creating sample data...")
        # Create sample data (16 rows for training)
        data = pd.DataFrame({
            'Survived': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Pclass': [3, 1, 3, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, 28.0, 54.0, 2.0, 30.0, 35.0, 19.0, 55.0, 60.0, 45.0, 32.0, 40.0, 20.0],
            'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 21.07, 26.55, 15.0, 10.0, 135.0, 12.0, 14.0, 60.0, 10.5, 7.0],
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'S', 'S', 'C', 'S', 'S', 'C', 'S', 'Q', 'C', 'S', 'S']
        })
        # Save the sample data for future use (if the real titanic.csv is missing)
        data.to_csv(DATA_PATH, index=False)
        return data

def train_and_save_model():
    """Trains a Logistic Regression model and saves it as a joblib file."""
    
    data = prepare_data()
    
    # Select features (X) and target (y)
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Data Pipeline (Preprocessing)
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Build the complete Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Save the model
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model successfully trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()