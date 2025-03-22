import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    try:
        # Load dataset
        df = pd.read_csv('house_prices_switzerland.csv')
        
        # Data cleaning
        # Drop unnecessary columns if any
        if 'size_units' in df.columns:
            df = df.drop(columns=['size_units'])
        
        # Handle missing values if any
        if df.isna().sum().any():
            df = df.dropna(subset=['Price'])  # Drop rows with missing prices
        
        # Feature engineering - calculate price per square meter
        if 'Price' in df.columns and 'LivingSpace' in df.columns:
            df['price_per_sqm'] = df['Price'] / df['LivingSpace']
        elif 'price' in df.columns and 'size' in df.columns:
            df['price_per_sqm'] = df['price'] / df['size']
        
        # Ensure we have consistent column names
        if 'HouseType' in df.columns:
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
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Fall back to the final_dataset.csv if house_prices_switzerland.csv fails
        return pd.read_csv('final_dataset.csv')