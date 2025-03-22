import pandas as pd
import pickle
import streamlit as st
import os
from data_processor import load_and_preprocess_data
from model_trainer import check_and_train_model

def load_model():
    """Load the trained model from disk."""
    # Check if model exists and train if needed
    model_exists = check_and_train_model()
    
    if model_exists:
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.warning("Training a new model...")
            from model_trainer import train_and_save_model
            df = load_and_preprocess_data()
            model, _ = train_and_save_model(df)
            return model
    return None

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
    
    if model is None:
        raise ValueError("Could not load or train model")
    
    # Create input DataFrame
    input_data = pd.DataFrame([[beds, baths, size, zip_code]],
                             columns=['beds', 'baths', 'size', 'zip_code'])
    
    # Ensure correct data types
    input_data['beds'] = input_data['beds'].astype(float)
    input_data['baths'] = input_data['baths'].astype(float)
    input_data['size'] = input_data['size'].astype(float)
    input_data['zip_code'] = input_data['zip_code'].astype(int)
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # Ensure prediction is positive and reasonable
        prediction = max(0, prediction)
        
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Fallback to a simple estimation based on size
        return size * 10000  # Simple fallback estimation