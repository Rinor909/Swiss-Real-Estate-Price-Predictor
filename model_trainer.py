import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def train_and_save_model(df):
    """
    Train the model and save it to disk.
    Returns the trained pipeline and evaluation metrics.
    """
    try:
        # Prepare data
        X = df.drop(columns=['price'])
        y = df['price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Set up preprocessing
        categorical_features = ['beds']  # Expand as needed
        numerical_features = ['baths', 'size', 'zip_code']
        
        categorical_transformer = OneHotEncoder(sparse=False)
        numerical_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ]
        )
        
        # Create pipeline with Ridge regression
        ridge_model = Ridge(alpha=1.0)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', ridge_model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return pipeline, metrics
    
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None