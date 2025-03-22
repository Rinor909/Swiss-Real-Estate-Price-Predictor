import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_dataset_info(df):
    """
    Display information about the dataset.
    
    Args:
        df: DataFrame to analyze
    """
    st.subheader("Dataset Overview")
    
    # Basic information
    st.write(f"Number of properties: {len(df)}")
    st.write(f"Number of features: {df.shape[1]}")
    
    # Price statistics
    st.subheader("Price Statistics")
    stats = df['price'].describe()
    stats_df = pd.DataFrame({
        'Statistic': stats.index,
        'Value (CHF)': stats.values
    })
    st.table(stats_df)
    
    # Dataset preview
    st.subheader("Dataset Sample")
    st.dataframe(df.head())

def generate_canton_from_zip(zip_code):
    """
    Generate approximate canton based on postal code.
    This is a simplified mapping function for Swiss postal codes.
    
    Args:
        zip_code: Postal code
        
    Returns:
        Canton name
    """
    zip_str = str(zip_code)
    if zip_str.startswith('1'):
        return 'Vaud/Geneva'
    elif zip_str.startswith('2'):
        return 'Neuchâtel/Jura'
    elif zip_str.startswith('3'):
        return 'Bern'
    elif zip_str.startswith('4'):
        return 'Basel'
    elif zip_str.startswith('5'):
        return 'Aargau'
    elif zip_str.startswith('6'):
        return 'Central Switzerland'
    elif zip_str.startswith('7'):
        return 'Graubünden'
    elif zip_str.startswith('8'):
        return 'Zürich'
    elif zip_str.startswith('9'):
        return 'Eastern Switzerland'
    else:
        return 'Unknown'

def create_price_per_sqm_chart(df):
    """Create a chart showing price per square meter by canton."""
    # Add canton column
    df['canton'] = df['zip_code'].apply(generate_canton_from_zip)
    
    # Calculate median price per square meter by canton
    canton_price_df = df.groupby('canton')['price_per_sqm'].median().reset_index()
    
    # Sort by price
    canton_price_df = canton_price_df.sort_values('price_per_sqm', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(canton_price_df['canton'], canton_price_df['price_per_sqm'])
    
    # Customize chart
    plt.title('Median Price per Square Meter by Canton')
    plt.xlabel('Canton')
    plt.ylabel('Price per m² (CHF)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig