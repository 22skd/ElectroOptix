import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_resource
def load_data():
    data = pd.read_csv('modified_excel_file.csv')
    return data

def train_model(data):
    X = data.drop(columns=['Date','Time','Energy (kWh)'])
    y = data['Energy (kWh)']
    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)
    return rf_model

def make_predictions(data, num_days, rf_model):
    X = data.drop(columns=['Date','Time','Energy (kWh)'])
    y = data['Energy (kWh)']
    
    # Generate future dates for predictions
    future_dates = pd.date_range(start=pd.Timestamp.now().date() + pd.Timedelta(days=1), periods=num_days)
    
    # Make predictions for the next num_days
    rf_predictions = rf_model.predict(X.iloc[:num_days]).round(2)
    
    # Get actual values for the last 20 days
    actual_values = y.tail(num_days)
    
    # Create DataFrame for predictions
    rf_predictions_df = pd.DataFrame(rf_predictions, columns=['Energy (kWh) Predictions'])
    
    # Create DataFrame for dates
    date_df = pd.DataFrame(future_dates, columns=['Date'])
    
    # Concatenate date and prediction DataFrames
    predictions_df = pd.concat([date_df, rf_predictions_df], axis=1)

    # Calculate evaluation metrics
    rf_mse = mean_squared_error(y.iloc[:num_days], rf_predictions)
    rf_r2 = r2_score(y.iloc[:num_days], rf_predictions)
    rf_mae = mean_absolute_error(y.iloc[:num_days], rf_predictions)
    rf_rmse = mean_squared_error(y.iloc[:num_days], rf_predictions, squared=False)
    rf_mape = np.mean(np.abs((y.iloc[:num_days] - rf_predictions) / y.iloc[:num_days]) * 100)

    rf_metrics = {'Mean Absolute Error': rf_mae, 'Root Mean Squared Error': rf_rmse, 'Mean Absolute Percentage Error': rf_mape}

    return predictions_df, rf_mse, rf_r2, rf_metrics, actual_values

def plot_predicted_values(predictions_df):

    plt.figure(figsize=(10, 6))
    plt.plot(predictions_df['Date'], predictions_df['Energy (kWh) Predictions'], label='Predicted Energy (kWh)', marker='o', color='red')
    plt.title('Predicted Energy (kWh)')
    plt.xlabel('Date')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

def plot_actual_vs_predicted(predictions_df, actual_values):

    plt.figure(figsize=(10, 6))
    plt.plot(predictions_df['Date'], predictions_df['Energy (kWh) Predictions'], label='Predicted Energy (kWh)', marker='o', color='red')
    plt.plot(predictions_df['Date'], actual_values, label='Actual Energy (kWh)', marker='o', color='blue')
    plt.title('Actual vs Predicted Energy (kWh)')
    plt.xlabel('Date')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

def main():
    st.title('Energy Prediction App')
    data = load_data()
    rf_model = train_model(data)
    num_days = st.number_input('Enter the number of days for prediction:', min_value=1, max_value=365)

    if st.button('Generate Predictions'):

        predictions_df, rf_mse, rf_r2, rf_metrics, actual_values = make_predictions(data, num_days, rf_model)

        st.subheader(f'Predictions for the next {num_days} days:')
        st.write(predictions_df)

        st.subheader('Evaluation Metrics:')
        st.write(f'Random Forest MSE: {rf_mse:.2f}')
        st.write(f'Random Forest R-squared: {rf_r2:.2f}')

        st.subheader('Metrics for Random Forest Model:')
        st.write(rf_metrics)

        st.subheader('Predictions Visualization')
        plot_predicted_values(predictions_df)
        st.subheader('Actual vs Predicted Energy (kWh)')
        plot_actual_vs_predicted(predictions_df, actual_values)

if __name__ == '__main__':
    main()
