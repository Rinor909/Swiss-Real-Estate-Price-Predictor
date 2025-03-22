# Swiss Real Estate Price Predictor

## Overview
This Swiss Real Estate Price Predictor is a Streamlit web application that allows users to estimate property prices in Switzerland based on key characteristics such as bedrooms, bathrooms, living space, and location. The application uses machine learning regression models to make accurate predictions based on real Swiss property data.

## Features
- **Price Prediction**: Input property details to get an instant price estimate
- **Data Visualization**: Explore property price distributions across different locations
- **Interactive Analysis**: Analyze how different factors affect property prices
- **Machine Learning Models**: Uses Ridge Regression for robust predictions

## Project Structure
```
├── app.py                # Main Streamlit application
├── data_processor.py     # Data loading and preprocessing
├── model_trainer.py      # ML model training and evaluation
├── predictor.py          # Price prediction functionality
├── visualizations.py     # Data visualization components
├── utils.py              # Utility functions
├── house_prices_switzerland.csv  # Primary dataset
├── model.pkl             # Saved trained model
└── requirements.txt      # Project dependencies
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/swiss-property-predictor.git
   cd swiss-property-predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. Open your web browser and navigate to `http://localhost:8501`

## Usage
1. Navigate to the "Price Prediction" tab
2. Enter property details:
   - Number of bedrooms
   - Number of bathrooms
   - Property size in square meters
   - Postal code
3. Click "Predict Price" to see the estimated property value
4. Explore data visualizations in the "Data Analysis" tab
5. Learn about the model in the "Model Information" tab

## Data
The application uses a dataset of Swiss property listings with the following features:
- Number of bedrooms
- Number of bathrooms
- Property size
- Postal code
- Price (target variable)

## Technical Details
- **Machine Learning**: Ridge Regression with hyperparameter tuning
- **Data Preprocessing**: Standardization, one-hot encoding
- **Visualization**: Interactive charts using Plotly
- **Web Framework**: Streamlit for UI/UX

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- Dataset provided by [data source]
- Project created as part of university coursework
- Built with Streamlit, scikit-learn, and Plotly

## Contact
For questions or feedback, please contact [your email or contact information]
