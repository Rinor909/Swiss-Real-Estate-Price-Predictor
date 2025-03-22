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
            try:
                # Load the CSV file directly without any type conversions yet
                df = pd.read_csv('house_prices_switzerland.csv', dtype={
                    'PostalCode': str,  # Keep as string initially
                    'ID': str,
                    'HouseType': str,
                    'Size': object,  # Could be S, M, L or numeric
                    'Price': float,
                    'LotSize': float,
                    'Balcony': object,
                    'LivingSpace': float,
                    'NumberRooms': float,
                    'YearBuilt': float,
                    'Locality': str
                })
                
                # Rename columns for consistency
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
                
                # Create default columns if missing
                if 'beds' not in df.columns:
                    df['beds'] = 3.0
                if 'baths' not in df.columns:
                    df['baths'] = df['beds'] / 2.5  # Rough estimate
                if 'size' not in df.columns and 'living_space' in df.columns:
                    df['size'] = df['living_space']
                elif 'size' not in df.columns:
                    df['size'] = 200.0
                if 'zip_code' not in df.columns:
                    df['zip_code'] = '1000'  # Default as string
                if 'price' not in df.columns:
                    df['price'] = 1000000.0
                
                # Convert size codes to numeric if needed
                if df['size'].dtype == object:
                    # Try to convert directly to numeric first
                    df['size'] = pd.to_numeric(df['size'], errors='coerce')
                    # Then check for S, M, L values
                    size_mapping = {'S': 150.0, 'M': 250.0, 'L': 350.0}
                    mask = df['size'].isna()
                    df.loc[mask, 'size'] = df.loc[mask, 'size'].map(size_mapping)
                
                # Fill missing values with reasonable defaults
                df['beds'] = pd.to_numeric(df['beds'], errors='coerce').fillna(3.0)
                df['baths'] = pd.to_numeric(df['baths'], errors='coerce').fillna(2.0)
                df['size'] = pd.to_numeric(df['size'], errors='coerce').fillna(200.0)
                df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(1000000.0)
                
                # Handle zip_code separately - keep as string initially to avoid NaN conversion issues
                if df['zip_code'].dtype != str:
                    df['zip_code'] = df['zip_code'].astype(str)
                df['zip_code'] = df['zip_code'].fillna('1000')
                # Clean non-numeric characters from zip codes
                df['zip_code'] = df['zip_code'].str.replace(r'[^0-9]', '', regex=True)
                # If empty after cleaning, use default
                df.loc[df['zip_code'] == '', 'zip_code'] = '1000'
                # Now we should be able to safely convert to numeric for modeling
                df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce').fillna(1000).astype(int)
                
                # Calculate price per square meter
                df['price_per_sqm'] = df['price'] / df['size']
                
                # Drop any rows where price or size is 0 or negative
                df = df[(df['price'] > 0) & (df['size'] > 0)]
                
                return df
                
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                raise
        else:
            raise FileNotFoundError("house_prices_switzerland.csv not found")
    
    except Exception as e:
        # Create a dummy dataset
        st.warning(f"Data loading error: {str(e)}. Using a dummy dataset for demonstration purposes.")
        dummy_data = {
            'beds': [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.0, 2.0, 4.0, 3.0],
            'baths': [2.0, 2.5, 1.0, 3.0, 2.0, 2.5, 2.0, 1.5, 2.5, 1.5],
            'size': [150.0, 200.0, 100.0, 250.0, 180.0, 220.0, 160.0, 120.0, 210.0, 170.0],
            'zip_code': [8001, 8002, 8003, 8004, 1000, 1001, 1002, 1003, 6900, 3000],
            'price': [800000.0, 1200000.0, 600000.0, 1500000.0, 700000.0, 1100000.0, 900000.0, 750000.0, 1300000.0, 850000.0]
        }
        df = pd.DataFrame(dummy_data)
        df['price_per_sqm'] = df['price'] / df['size']
        return df