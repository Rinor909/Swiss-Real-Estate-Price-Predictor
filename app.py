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
    # App configuration
    st.set_page_config(
        page_title="Swiss Real Estate Price Predictor",
        page_icon="🏡",
        layout="wide"
    )
    
    # Header
    st.title("🏡 Swiss Real Estate Price Predictor")
    
    try:
        # Load data
        with st.spinner("Loading property data..."):
            df = load_and_preprocess_data()
        
        # Make sure all required columns exist
        required_columns = ['beds', 'baths', 'size', 'zip_code', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Please check your data source or update the data processing logic.")
            return
        
        # App tabs
        tab1, tab2, tab3 = st.tabs(["📊 Price Prediction", "📈 Data Analysis", "ℹ️ Model Information"])
        
        with tab1:
            st.header("Predict Property Price")
            
            # Create two columns for input parameters
            col1, col2 = st.columns(2)
            
            try:
                with col1:
                    st.subheader("Property Characteristics")
                    # User inputs for prediction with proper error handling
                    beds_min = int(df['beds'].min()) if pd.notna(df['beds'].min()) else 1
                    beds_max = int(df['beds'].max()) if pd.notna(df['beds'].max()) else 10
                    # Make sure min and max are different
                    if beds_min == beds_max:
                        beds_max = beds_min + 1
                    beds = st.slider("Number of Bedrooms", beds_min, beds_max, min(beds_min + 2, beds_max), step=1)
                    
                    # Create a whole number bathroom selector
                    baths_min = int(df['baths'].min()) if pd.notna(df['baths'].min()) else 1
                    baths_max = int(df['baths'].max()) if pd.notna(df['baths'].max()) else 5
                    
                    # Make sure min and max are different
                    if baths_min == baths_max:
                        baths_max = baths_min + 1
                    
                    # Use whole numbers only
                    baths_default = min(baths_min + 1, baths_max)
                    
                    baths = st.slider("Number of Bathrooms", 
                                     min_value=int(baths_min), 
                                     max_value=int(baths_max), 
                                     value=int(baths_default),
                                     step=1)  # Step of 1 bathroom (whole numbers only)
                
                with col2:
                    st.subheader("Property Size & Location")
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
                        # Get most common zip codes for better UX
                        top_zips = df['zip_code'].value_counts().head(10).index.tolist()
                        # Add options for different regions
                        zip_options = {
                            'Zürich Area (80xx)': [8001, 8002, 8003, 8004, 8005],
                            'Geneva Area (12xx)': [1201, 1202, 1203, 1204, 1205],
                            'Basel Area (40xx)': [4001, 4051, 4052, 4053, 4054],
                            'Bern Area (30xx)': [3001, 3004, 3005, 3006, 3007],
                            'Lugano Area (69xx)': [6900, 6901, 6902, 6903, 6904],
                            'Other': sorted(df['zip_code'].unique().tolist()[:20])
                        }
                        
                        # Two step selection for better UX
                        region = st.selectbox("Select Region", list(zip_options.keys()))
                        zip_code = st.selectbox("Select Postal Code", 
                                              zip_options[region] if region != 'Other' else sorted(top_zips))
                    else:
                        zip_code = st.number_input("Postal Code", min_value=1000, max_value=9999, value=8001)
                
                # Add a divider
                st.divider()
                
                # Center the prediction button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    predict_button = st.button("🔍 Predict Property Price", use_container_width=True)
                
                # Display prediction result
                if predict_button:
                    with st.spinner('Calculating property price...'):
                        try:
                            price = predict_price(beds, baths, size, zip_code)
                            
                            # Use Streamlit's metrics for a nice display
                            st.success(f"Estimated property price: CHF {price:,.2f}")
                            
                            # Property summary
                            st.write(f"Based on {beds} bedrooms, {baths} bathrooms, {size}m², in postal code {zip_code}")
                            
                            # Show additional context
                            with st.expander("Price Context"):
                                avg_price = df['price'].mean()
                                median_price = df['price'].median()
                                
                                # Use metrics for a nice display
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Your Estimate", f"CHF {price:,.0f}")
                                with col2:
                                    st.metric("Avg Price in Switzerland", f"CHF {avg_price:,.0f}")
                                with col3:
                                    st.metric("Median Price in Switzerland", f"CHF {median_price:,.0f}")
                                
                                # Calculate price per m²
                                price_per_sqm = price / size
                                avg_price_per_sqm = df['price'].mean() / df['size'].mean()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Price per m²", f"CHF {price_per_sqm:,.0f}")
                                with col2:
                                    st.metric("Avg Price per m²", f"CHF {avg_price_per_sqm:,.0f}")
                                
                                # Compare to similar properties
                                similar = df[
                                    (df['beds'] == beds) & 
                                    (df['baths'] >= baths - 0.5) & 
                                    (df['baths'] <= baths + 0.5) & 
                                    (df['size'] >= size - 30) & 
                                    (df['size'] <= size + 30)
                                ]
                                
                                if len(similar) > 0:
                                    similar_avg = similar['price'].mean()
                                    st.metric("Similar Properties Avg", f"CHF {similar_avg:,.0f}", 
                                             delta=f"{(price-similar_avg)/similar_avg*100:.1f}%")
                                    
                                    if price > similar_avg * 1.2:
                                        st.warning("This estimate is above average for similar properties")
                                    elif price < similar_avg * 0.8:
                                        st.success("This estimate is below average for similar properties")
                                    else:
                                        st.info("This estimate is in line with similar properties")
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
                            st.info("Please try with different input values or check the model.")
            except Exception as e:
                st.error(f"Error setting up input sliders: {e}")
                st.info("There might be issues with the data ranges. Try refreshing or check the data.")
        
        with tab2:
            st.header("Data Analysis")
            
            try:
                # Add tabs within the Data Analysis tab for organization
                analysis_tabs = st.tabs(["Price Distribution", "Price Heatmap", "Price Factors"])
                
                with analysis_tabs[0]:
                    st.subheader("Price Distribution by Postal Code")
                    st.info("This chart shows median property prices in the top postal code areas")
                    price_dist_chart = plot_price_distribution(df)
                    st.plotly_chart(price_dist_chart, use_container_width=True)
                
                with analysis_tabs[1]:
                    st.subheader("Property Price Heatmap")
                    st.info("This heatmap illustrates how property prices vary by size and number of bedrooms")
                    heatmap_chart = plot_price_heatmap(df)
                    st.plotly_chart(heatmap_chart, use_container_width=True)
                
                with analysis_tabs[2]:
                    # Add some custom metrics
                    st.subheader("Key Property Price Factors")
                    
                    # Price stats by property size
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("##### Price by Property Size")
                        # Create size categories
                        df['size_category'] = pd.cut(
                            df['size'], 
                            bins=[0, 100, 200, 300, 500, 1000, float('inf')],
                            labels=['< 100m²', '100-200m²', '200-300m²', '300-500m²', '500-1000m²', '> 1000m²']
                        )
                        size_stats = df.groupby('size_category')['price'].median().reset_index()
                        st.bar_chart(size_stats.set_index('size_category'))
                    
                    with col2:
                        st.write("##### Price by Number of Bedrooms")
                        bed_stats = df.groupby('beds')['price'].median().reset_index()
                        # Only keep reasonable bedroom numbers for the chart
                        bed_stats = bed_stats[(bed_stats['beds'] >= 1) & (bed_stats['beds'] <= 10)]
                        st.bar_chart(bed_stats.set_index('beds'))
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")
                st.info("There might be issues with the data format. Check the data structure.")
        
        with tab3:
            st.header("Model Information")
            st.write("This model uses Ridge Regression to predict Swiss property prices.")
            
            # Create a more organized layout
            col1, col2 = st.columns(2)
            
            try:
                with col1:
                    st.subheader("Feature Importance")
                    st.info("This chart shows the relative importance of each feature in predicting property prices")
                    feature_chart = plot_feature_importance()
                    st.plotly_chart(feature_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Model Accuracy")
                    st.info("This chart compares predicted prices to actual prices to show model accuracy")
                    accuracy_chart = plot_actual_vs_predicted(df)
                    st.plotly_chart(accuracy_chart, use_container_width=True)
                
                # Add model details in an expander
                with st.expander("Model Details"):
                    st.header("Ridge Regression Model")
                    
                    st.write("""
                    This application uses a Ridge Regression model, which is a type of linear regression that includes 
                    L2 regularization to prevent overfitting. The model was trained on Swiss property data with the 
                    following features:
                    """)
                    
                    st.write("- Number of bedrooms")
                    st.write("- Number of bathrooms")
                    st.write("- Property size (m²)")
                    st.write("- Postal code (location)")
                    
                    st.subheader("Data Preprocessing")
                    
                    st.write("""
                    Before training, the data underwent several preprocessing steps:
                    
                    1. Missing values were handled
                    2. Categorical features were one-hot encoded
                    3. Numerical features were standardized
                    4. Outliers were identified and addressed
                    """)
                    
                    st.subheader("Model Evaluation")
                    
                    st.write("""
                    The model was evaluated using cross-validation with the following metrics:
                    
                    - R² Score: How much of the price variation the model explains
                    - RMSE (Root Mean Squared Error): The average prediction error in CHF
                    
                    These metrics help us understand how well the model performs on unseen data.
                    """)
            except Exception as e:
                st.error(f"Error displaying model information: {e}")
                st.info("The model might not be trained yet or there's an issue with the visualizations.")
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your data sources and make sure all dependencies are installed.")
        
    # Add footer with additional information
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.caption("Swiss Real Estate Price Predictor | Data last updated: March 2025")
        st.caption("Built with Streamlit, scikit-learn, and Plotly")

if __name__ == "__main__":
    main()