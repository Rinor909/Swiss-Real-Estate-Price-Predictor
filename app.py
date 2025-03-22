import streamlit as st
import pandas as pd
import numpy as np
from data_processor import load_and_preprocess_data
from predictor import predict_price
from visualizations import (
    plot_price_distribution, 
    plot_feature_importance,
    plot_price_heatmap,
    plot_actual_vs_predicted
)

def main():
    st.title("Swiss Real Estate Price Predictor")
    
    # Sidebar
    st.sidebar.header("Property Details")
    
    # Load data
    df = load_and_preprocess_data()
    
    # App tabs
    tab1, tab2, tab3 = st.tabs(["Price Prediction", "Data Analysis", "Model Information"])
    
    with tab1:
        st.header("Predict Property Price")
        
        # User inputs for prediction
        beds = st.slider("Number of Bedrooms", int(df['beds'].min()), int(df['beds'].max()), 3)
        baths = st.slider("Number of Bathrooms", float(df['baths'].min()), float(df['baths'].max()), 2.0)
        size = st.slider("Size (m²)", int(df['size'].min()), int(df['size'].max()), 1500)
        zip_code = st.selectbox("Postal Code", sorted(df['zip_code'].unique()))
        
        if st.button("Predict Price"):
            price = predict_price(beds, baths, size, zip_code)
            st.success(f"The estimated property price is: CHF {price:,.2f}")
    
    with tab2:
        st.header("Data Analysis")
        
        # Display visualizations
        st.subheader("Price Distribution by Postal Code")
        st.plotly_chart(plot_price_distribution(df))
        
        st.subheader("Property Price Heatmap")
        st.plotly_chart(plot_price_heatmap(df))
    
    with tab3:
        st.header("Model Information")
        st.write("This model uses Ridge Regression to predict Swiss property prices.")
        
        st.subheader("Feature Importance")
        st.plotly_chart(plot_feature_importance())
        
        st.subheader("Model Accuracy")
        st.plotly_chart(plot_actual_vs_predicted(df))

if __name__ == "__main__":
    main()