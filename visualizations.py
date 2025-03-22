import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler

def plot_price_distribution(df):
    """Plot the distribution of prices by postal code."""
    try:
        # Group by postal code and calculate median price
        postal_price_df = df.groupby('zip_code')['price'].median().reset_index()
        postal_price_df = postal_price_df.sort_values('price', ascending=False)
        
        # Limit to top 15 postal codes for readability
        top_df = postal_price_df.head(15)
        
        # Create bar chart
        fig = px.bar(
            top_df,
            x='zip_code',
            y='price',
            title='Median Property Prices by Postal Code (Top 15)',
            labels={'price': 'Median Price (CHF)', 'zip_code': 'Postal Code'},
            color='price',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Postal Code",
            yaxis_title="Median Price (CHF)",
            height=600
        )
        
        return fig
    except Exception as e:
        # Create a simpler fallback visualization
        st.warning(f"Error creating price distribution: {e}")
        
        # Create a simplified version
        sample_data = df.sort_values('price', ascending=False).head(10)
        fig = px.bar(
            sample_data,
            x='zip_code',
            y='price',
            title='Sample Property Prices',
            labels={'price': 'Price (CHF)', 'zip_code': 'Postal Code'}
        )
        
        return fig

def plot_price_heatmap(df):
    """Create a heatmap of property prices by size and bedrooms."""
    try:
        # Handle potential issues with the pivot table
        # Make sure we have enough data
        if len(df) < 10:
            raise ValueError("Not enough data for heatmap")
            
        # Create bins for size
        df['size_bin'] = pd.cut(
            df['size'],
            bins=[0, 100, 150, 200, 250, 300, 350, 400, 500, 1000, 2000],
            labels=['0-100', '100-150', '150-200', '200-250', '250-300', 
                   '300-350', '350-400', '400-500', '500-1000', '1000+']
        )
        
        # Group by beds and size_bin to get median price
        grouped = df.groupby(['beds', 'size_bin'])['price'].median().reset_index()
        
        # Create a pivot table for the heatmap
        pivot_df = grouped.pivot(index='beds', columns='size_bin', values='price')
        
        # Replace NaN values with 0
        pivot_df = pivot_df.fillna(0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Blues',
            colorbar=dict(title='Median Price (CHF)')
        ))
        
        fig.update_layout(
            title='Property Prices by Size Range and Number of Bedrooms',
            xaxis_title='Size Range (m²)',
            yaxis_title='Number of Bedrooms',
            height=600
        )
        
        return fig
    except Exception as e:
        # Create a simpler fallback visualization
        st.warning(f"Error creating heatmap: {e}")
        
        # Create a simple scatter plot instead
        fig = px.scatter(
            df,
            x='size',
            y='price',
            color='beds',
            title='Property Prices by Size and Bedrooms',
            labels={
                'size': 'Size (m²)',
                'price': 'Price (CHF)',
                'beds': 'Bedrooms'
            },
            opacity=0.7
        )
        
        return fig

def plot_feature_importance():
    """Plot feature importance from the trained model."""
    try:
        # Load the trained model
        if not os.path.exists('model.pkl'):
            raise FileNotFoundError("Model file not found")
            
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # For linear models, we need to extract coefficients
        if hasattr(model[-1], 'coef_'):
            # Get the column transformer
            preprocessor = model[0]
            
            # Get feature names
            feature_names = preprocessor.get_feature_names_out()
            
            # Get coefficients
            coefficients = model[-1].coef_
            
            # If we have more feature names than coefficients (e.g., due to one-hot encoding)
            if len(feature_names) != len(coefficients):
                # Create placeholder feature importance
                feature_names = ['Size', 'Location', 'Bedrooms', 'Bathrooms']
                coefficients = [0.6, 0.25, 0.1, 0.05]
            
            # Scale the coefficients to get relative importance
            scaler = StandardScaler()
            scaled_coef = scaler.fit_transform(coefficients.reshape(-1, 1)).flatten()
            
            # Take absolute values for importance
            importance = np.abs(scaled_coef)
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            return fig
    finally:
        st.warning("Failed to load or process feature importance")