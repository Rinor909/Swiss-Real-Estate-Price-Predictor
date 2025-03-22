import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    try:
        # Check if file exists
        if os.path.exists('house_prices_switzerland.csv'):
            # Load dataset
            df = pd.read_csv('house_prices_switzerland.csv')
            
            # Data cleaning
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
            
            # Handle missing values
            df = df.dropna(subset=['price'], how='all')  # Drop rows with missing prices
            
            # Convert size codes to approximate square meters if needed
            if 'size' in df.columns and df['size'].dtype == 'object':
                size_mapping = {'S': 150, 'M': 250, 'L': 350}
                df['size'] = df['size'].map(size_mapping)
            
            # Use living_space as size if available and size is missing
            if 'living_space' in df.columns:
                df['size'] = df['size'].fillna(df['living_space'])
            
            # Add baths column if it doesn't exist (using a placeholder calculation)
            if 'baths' not in df.columns:
                # Typically, a property has about 1 bathroom per 2-3 bedrooms
                # This is a rough estimate - adjust as needed for Swiss properties
                df['baths'] = (df['beds'] / 2.5).apply(lambda x: max(1.0, round(x * 2) / 2))
                st.info("Note: Bathroom data was estimated based on number of bedrooms.")
            
            # Feature engineering - calculate price per square meter
            if 'price' in df.columns and 'size' in df.columns:
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
            df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
            df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
            df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce').astype(int)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            return df
        else:
            # If file doesn't exist, create a dummy dataset
            raise FileNotFoundError("house_prices_switzerland.csv not found")
    
    except Exception as e:
        # Create a dummy dataset with the required columns
        st.warning(f"Data loading error: {e}. Using a dummy dataset for demonstration purposes.")
        dummy_data = {
            'beds': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3, 5, 2, 3, 4],
            'baths': [2.0, 2.5, 1.0, 3.0, 2.0, 2.5, 2.0, 1.5, 2.5, 1.5, 3.0, 1.0, 2.0, 2.5],
            'size': [150, 200, 100, 250, 180, 220, 160, 120, 210, 170, 260, 110, 190, 230],
            'zip_code': [8001, 8002, 8003, 8004, 1000, 1001, 1002, 1003, 6900, 3000, 4000, 9000, 7000, 5000],
            'price': [800000, 1200000, 600000, 1500000, 700000, 1100000, 900000, 750000, 1300000, 850000, 1600000, 650000, 950000, 1400000]
        }
        df = pd.DataFrame(dummy_data)
        
        # Add price per square meter
        df['price_per_sqm'] = df['price'] / df['size']
        
        return df