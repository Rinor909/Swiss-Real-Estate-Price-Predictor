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
            df = df.dropna(subset=['Price'], how='all')  # Drop rows with missing prices
        
        # Ensure we have consistent column names
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
        
        # Add baths column if it doesn't exist (using a placeholder calculation)
        if 'baths' not in df.columns:
            # Typically, a property has about 1 bathroom per 2-3 bedrooms
            # This is a rough estimate - adjust as needed for Swiss properties
            df['baths'] = (df['beds'] / 2.5).apply(lambda x: max(1.0, round(x * 2) / 2))
            st.info("Note: Bathroom data was estimated based on number of bedrooms.")
        
        # Feature engineering - calculate price per square meter
        if 'price' in df.columns and 'living_space' in df.columns:
            df['price_per_sqm'] = df['price'] / df['living_space']
        elif 'price' in df.columns and 'size' in df.columns:
            df['price_per_sqm'] = df['price'] / df['size']
        
        # Ensure all necessary columns exist and handle missing values
        required_columns = ['beds', 'baths', 'size', 'zip_code', 'price']
        for col in required_columns:
            if col not in df.columns:
                if col == 'size' and 'living_space' in df.columns:
                    df['size'] = df['living_space']
                else:
                    st.warning(f"Column {col} is missing. Using placeholder data.")
                    if col == 'baths':
                        df['baths'] = 1.0
                    elif col == 'beds':
                        df['beds'] = 3
                    elif col == 'size':
                        df['size'] = 200
                    elif col == 'zip_code':
                        df['zip_code'] = 1000
                    elif col == 'price':
                        df['price'] = 1000000
        
        # Drop rows with null values in required columns
        df = df.dropna(subset=required_columns)
        
        # Make sure all columns have the correct data types
        df['beds'] = df['beds'].astype(float)
        df['baths'] = df['baths'].astype(float)
        df['size'] = df['size'].astype(float)
        df['zip_code'] = df['zip_code'].astype(int)
        df['price'] = df['price'].astype(float)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a dummy dataset with the required columns
        st.warning("Using a dummy dataset for demonstration purposes.")
        dummy_data = {
            'beds': [3, 4, 2, 5, 3, 4, 3, 2],
            'baths': [2.0, 2.5, 1.0, 3.0, 2.0, 2.5, 2.0, 1.5],
            'size': [150, 200, 100, 250, 180, 220, 160, 120],
            'zip_code': [8001, 8002, 8003, 8004, 1000, 1001, 1002, 1003],
            'price': [800000, 1200000, 600000, 1500000, 700000, 1100000, 900000, 750000]
        }
        return pd.DataFrame(dummy_data)