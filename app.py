import streamlit as st
from pathlib import Path
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as pl
import numpy as np
from scipy.interpolate import interp1d

st.title('Agricultural Commodity Price Projection')
st.sidebar.info('Welcome to the Agricultural Commodity Price Projection App. Choose your options below')

# Sidebar options
option = st.sidebar.selectbox('Select Crop', ['Maize (new harvest)', 'Beans'])
start_date = st.sidebar.date_input('Start Date', value=datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input('End Date', datetime.date.today())
num_days_forecast = st.sidebar.number_input('Number of days to forecast', value=7, min_value=1, max_value=365, step=1)
selected_district = st.sidebar.selectbox('Select District', ['Dedza', 'Mzimba', 'Blantyre', 'Ntcheu'])

# Filter markets based on selected district
if selected_district == 'Dedza':
    markets = ['Lizulu', 'Nsikawanjala']
elif selected_district == 'Mzimba':
    markets = ['Jenda']
elif selected_district == 'Blantyre':
    markets = ['Lunzu']
elif selected_district == 'Ntcheu':
    markets = ['Market1', 'Market2']  # Add the markets for Ntcheu here

selected_market = st.sidebar.selectbox('Select Market', markets)
forecast_button = st.sidebar.button('Predict')

# Define the path to the CSV file
DATA_PATH = Path.cwd() / 'data' / 'wfp_food_prices_mwi.csv'

# Read the data from the CSV file
data = pd.read_csv(DATA_PATH)

# Display the raw data using Streamlit
st.subheader("Raw WFP Data")
st.write(data)

# Remove null values by replacing with bfill
ft_data = data.fillna('bfill', inplace=False)
ft_data.isnull().sum()

# Display the filtered data after filling null values
st.subheader("Filtered WFP Data (Nulls Filled)")
st.write(ft_data)

# Drop the specified columns
columns_to_drop = ['usdprice', 'latitude', 'longitude', 'category', 'unit', 'priceflag', 'currency', 'pricetype']
ft_data.drop(columns=columns_to_drop, inplace=True)
ft_data.drop(index=0, inplace=True)

# Display the data after dropping columns
st.subheader('Filtered Data After Dropping Columns')
st.write(ft_data)

# Filter data based on the date, commodity, district, and market
# Converting the date column to datetime format
ft_data['date'] = pd.to_datetime(ft_data['date'])

# Defining the date range
start_dates = start_date.strftime('%Y-%m-%d')
end_dates = end_date.strftime('%Y-%m-%d')

# Filtering the data for the date range, commodity, district, and market
filtered_df = ft_data[(ft_data['date'] >= start_dates) & (ft_data['date'] <= end_dates)]
filtered_df = filtered_df[(filtered_df['commodity'] == option) & (filtered_df['district'] == selected_district) & (filtered_df['market'] == selected_market)]

# Display the fully filtered data
st.subheader('Fully Filtered Data')
st.write(filtered_df)

# Generate trend graph for the fully filtered data
if not filtered_df.empty:
    fig, ax = plt.subplots()
    ax.plot(filtered_df['date'], filtered_df['price'], label='Historical Prices', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Historical Prices Trend')
    ax.legend()
    st.pyplot(fig)

if forecast_button:
    if len(filtered_df) < 2:
        st.error("Insufficient data to make predictions. Please adjust your filtering criteria.")
    else:
        # Filter the DataFrame based on the selected crop
        filtered_data = filtered_df[filtered_df['commodity'] == option]

        # Check if any data is available for the selected crop
        if filtered_data.empty:
            st.warning(f"No data available for {option}. Please adjust your filtering criteria.")
        else:
            # Get the prices for the selected crop
            y = filtered_data['price'].values

            # Preprocessing the Features
            # Convert categorical features into numerical representations (e.g., one-hot encoding)
            encoded_districts = pd.get_dummies(filtered_df['district'], prefix='district')
            encoded_markets = pd.get_dummies(filtered_df['market'], prefix='market')

            # Concatenate the numerical representation of features
            X_date = filtered_df[['date']].values.astype(int)
            X = np.concatenate([X_date, encoded_districts, encoded_markets], axis=1)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model training
            regressor = RandomForestRegressor()
            regressor.fit(X_train, y_train)

            # Model prediction
            y_pred = regressor.predict(X_test)

            # Evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display evaluation metrics
            st.write(f'Mean Squared Error: {mse}')
            st.write(f'R^2 Score: {r2}')

            # Forecasting for multiple days
            forecast_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=num_days_forecast)  # Change the number of periods as needed
            forecast_features = filtered_df[['date']].tail(1).values.astype(int)  # Use the last available date

            # Concatenate the last available date with encoded districts and markets for forecasting
            forecast_districts = pd.get_dummies(filtered_df['district'].tail(1), prefix='district')
            forecast_markets = pd.get_dummies(filtered_df['market'].tail(1), prefix='market')

            # Check the number of features expected by the model
            expected_num_features = len(X[0])
            num_missing_features = expected_num_features - (len(forecast_districts.columns) + len(forecast_markets.columns) + 1)  # Subtract 1 for the date feature

            # Pad with zeros if the number of features is less than expected
            if num_missing_features > 0:
                zeros_to_pad = np.zeros((1, num_missing_features))
                forecast_features = np.concatenate([forecast_features, zeros_to_pad], axis=1)

            forecast_features = np.concatenate([forecast_features, forecast_districts.values, forecast_markets.values], axis=1)

            forecast_prices = []
            for _ in range(num_days_forecast):  
                forecast_price = regressor.predict(forecast_features)
                forecast_prices.append(forecast_price[0])
                # Update the date for the next forecast (you might need to adjust this based on your dataset)
                forecast_features[0][0] += 1

            # Create DataFrame for forecasted prices
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecasted Price': forecast_prices
            })

            st.subheader('Forecasted Prices')
            st.write(forecast_df)

            # Plot historical and forecasted prices using a line plot
            fig, ax = plt.subplots()

            ax.plot(filtered_df['date'], filtered_df['price'], label='Historical Prices', marker='o')

            ax.plot(forecast_dates, forecast_prices, label='Forecasted Prices', marker='o')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Historical and Forecasted Prices')
            ax.legend()
            st.pyplot(fig)
