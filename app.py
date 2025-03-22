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
    
    try:
        # Load data
        df = load_and_preprocess_data()
        
        # For debugging: Show available columns
        # st.write("Available columns:", df.columns.tolist())
        
        # Make sure all required columns exist
        required_columns = ['beds', 'baths', 'size', 'zip_code', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Please check your data source or update the data processing logic.")
            return
        
        # App tabs
        tab1, tab2, tab3 = st.tabs(["Price Prediction", "Data Analysis", "Model Information"])
        
        with tab1:
            st.header("Predict Property Price")
            
            try:
                # User inputs for prediction with proper error handling
                beds_min = int(df['beds'].min()) if pd.notna(df['beds'].min()) else 1
                beds_max = int(df['beds'].max()) if pd.notna(df['beds'].max()) else 10
                # Make sure min and max are different
                if beds_min == beds_max:
                    beds_max = beds_min + 1
                beds = st.slider("Number of Bedrooms", beds_min, beds_max, min(beds_min + 2, beds_max), step=1)
                
                # Create a more discrete bathroom selector similar to bedrooms
                baths_min = float(df['baths'].min()) if pd.notna(df['baths'].min()) else 1.0
                baths_max = float(df['baths'].max()) if pd.notna(df['baths'].max()) else 5.0
                # Make sure min and max are different
                if baths_min == baths_max:
                    baths_max = baths_min + 1.0
                
                # Round the bathroom values to 0.5 steps for more usable selection
                baths_min = round(baths_min * 2) / 2  # Round to nearest 0.5
                baths_max = round(baths_max * 2) / 2
                baths_default = round(min(baths_min + 1.0, baths_max) * 2) / 2
                
                baths = st.slider("Number of Bathrooms", 
                                 min_value=float(baths_min), 
                                 max_value=float(baths_max), 
                                 value=float(baths_default),
                                 step=0.5)  # Step of 0.5 bathrooms
                
                # Size in 10m² increments
                size_min = int(df['size'].min()) if pd.notna(df['size'].min()) else 50
                size_max = int(df['size'].max()) if pd.notna(df['size'].min()) else 500
                
                # Round to nearest 10m²
                size_min = (size_min // 10) * 10
                size_max = ((size_max + 9) // 10) * 10  # Round up to nearest 10
                
                # Make sure min and max are different
                if size_min == size_max:
                    size_max = size_min + 100
                
                size_default = min(size_min + 100, size_max)
                size_default = (size_default // 10) * 10  # Round to nearest 10m²
                
                size = st.slider("Size (m²)", 
                               min_value=int(size_min),
                               max_value=int(size_max),
                               value=int(size_default),
                               step=10)  # 10m² increments
                
                # Make sure we have zip codes to select from
                if df['zip_code'].nunique() > 0:
                    zip_code = st.selectbox("Postal Code", sorted(df['zip_code'].unique()))
                else:
                    zip_code = st.number_input("Postal Code", min_value=1000, max_value=9999, value=8001)
            
                if st.button("Predict Price"):
                    try:
                        price = predict_price(beds, baths, size, zip_code)
                        st.success(f"The estimated property price is: CHF {price:,.2f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                        st.info("Please try with different input values or check the model.")
            except Exception as e:
                st.error(f"Error setting up input sliders: {e}")
                st.info("There might be issues with the data ranges. Try refreshing or check the data.")
        
        with tab2:
            st.header("Data Analysis")
            
            try:
                # Display visualizations
                st.subheader("Price Distribution by Postal Code")
                price_dist_chart = plot_price_distribution(df)
                st.plotly_chart(price_dist_chart)
                
                st.subheader("Property Price Heatmap")
                heatmap_chart = plot_price_heatmap(df)
                st.plotly_chart(heatmap_chart)
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")
                st.info("There might be issues with the data format. Check the data structure.")
        
        with tab3:
            st.header("Model Information")
            st.write("This model uses Ridge Regression to predict Swiss property prices.")
            
            try:
                st.subheader("Feature Importance")
                feature_chart = plot_feature_importance()
                st.plotly_chart(feature_chart)
                
                st.subheader("Model Accuracy")
                accuracy_chart = plot_actual_vs_predicted(df)
                st.plotly_chart(accuracy_chart)
            except Exception as e:
                st.error(f"Error displaying model information: {e}")
                st.info("The model might not be trained yet or there's an issue with the visualizations.")
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your data sources and make sure all dependencies are installed.")

if __name__ == "__main__":
    main()