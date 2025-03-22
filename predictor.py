import pandas as pd
import pickle
import streamlit as st
from data_processor import load_and_preprocess_data

def load_model():
    """Load the trained model from disk."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Training a new model...")
        from model_trainer import train_and_save_model
        df = load_and_preprocess_data()
        model, _ = train_and_save_model(df)
        return model

def predict_price(beds, baths, size, zip_code):
    """
    Predict the price of a property.
    
    Args:
        beds: Number of bedrooms
        baths: Number of bathrooms
        size: Property size
        zip_code: Property postal code
        
    Returns:
        Predicted price
    """
    # Load model
    model = load_model()
    
    # Create input DataFrame
    input_data = pd.DataFrame([[beds, baths, size, zip_code]],
                             columns=['beds', 'baths', 'size', 'zip_code'])
    
    # Ensure correct data types
    input_data['beds'] = input_data['beds'].astype(int)
    input_data['baths'] = input_data['baths'].astype(float)
    input_data['size'] = input_data['size'].astype(float)
    input_data['zip_code'] = input_data['zip_code'].astype(int)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction