import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save_model(csv_filename):
    """
    Train a Ridge regression model on the Swiss property data and save it to disk.
    
    Args:
        csv_filename: Path to the CSV file with the property data
    """
    print(f"Loading data from {csv_filename}...")
    # Load the dataset
    df = pd.read_csv(csv_filename)
    
    # Basic preprocessing
    print("Preprocessing data...")
    # Check if we're working with the house_prices_switzerland.csv format
    if 'HouseType' in df.columns:
        # Rename columns to match expected format
        column_mapping = {
            'ID': 'id',
            'HouseType': 'house_type',
            'Size': 'size',
            'Price': 'price',
            'LotSize': 'lot_size',
            'Balcony': 'balcony',
            'LivingSpace': 'living_space',
            'NumberRooms': 'beds',
            'YearBuilt': 'year_built',
            'Locality': 'locality',
            'PostalCode': 'zip_code'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Make sure we only keep features we need
    required_columns = ['beds', 'baths', 'size', 'zip_code', 'price']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        return False
    
    # Select only the required columns
    df = df[required_columns]
    
    # Convert all to numeric just to be safe
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    print(f"Processed dataset shape: {df.shape}")
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    
    # Define preprocessing for numerical and categorical features
    numerical_features = ['baths', 'size', 'zip_code']
    categorical_features = ['beds']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create a pipeline with Ridge regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Model performance:")
    print(f"R² score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as model.pkl")
    return True

if __name__ == "__main__":
    # Try with different possible filenames
    possible_filenames = [
        'house_prices_switzerland.csv', 
        'final_dataset.csv'
    ]
    
    success = False
    for filename in possible_filenames:
        try:
            success = train_and_save_model(filename)
            if success:
                break
        except FileNotFoundError:
            print(f"File {filename} not found, trying next option...")
    
    if not success:
        print("Failed to create model. Please make sure one of the dataset files exists.")