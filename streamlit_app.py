# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Merged_Train.csv")

# Feature engineering: calculate temperature change and flag braking points
df['Temperature_Change'] = df['Temperature'].diff()
temperature_threshold = -10  # Threshold for significant temperature drop
df['BrakesApplied'] = np.where(df['Temperature_Change'] <= temperature_threshold, 1, 0)
df['CriticalTemperature'] = df.apply(lambda row: row['Temperature'] if row['BrakesApplied'] == 1 else np.nan, axis=1)

# Prepare the data for training
features = ['time', 'Frequency(Kmph)', 'Weight(Kg)']
target = 'Temperature'
X = df[features].dropna()
y = df[target].dropna()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
# Random Forest
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=42)
regressor_rf.fit(X_train, y_train)

# Decision Tree
regressor_dt = DecisionTreeRegressor(random_state=42)
regressor_dt.fit(X_train, y_train)

# Linear Regression
regressor_lr = LinearRegression()
regressor_lr.fit(X_train, y_train)

# Streamlit app layout
st.title("Brake Application Prediction")
st.write("This app predicts the critical temperature for applying brakes based on time, frequency, and weight.")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    time = st.sidebar.slider("Time (Seconds)", 0, 200, 100)
    frequency = st.sidebar.slider("Frequency (Kmph)", 0, 150, 50)
    weight = st.sidebar.slider("Weight (Kg)", 0, 200, 100)
    data = {
        'time': time,
        'Frequency(Kmph)': frequency,
        'Weight(Kg)': weight
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input parameters
st.subheader("Input Parameters")
st.write(input_df)

# Make predictions
rf_prediction = regressor_rf.predict(input_df)[0]
dt_prediction = regressor_dt.predict(input_df)[0]
lr_prediction = regressor_lr.predict(input_df)[0]

# Display predictions
st.subheader("Predicted Critical Temperature (°C) for Applying Brakes")
st.write(f"**Random Forest Prediction:** {rf_prediction:.2f}")
st.write(f"**Decision Tree Prediction:** {dt_prediction:.2f}")
st.write(f"**Linear Regression Prediction:** {lr_prediction:.2f}")

# Visualization section
st.subheader("Visualize Prediction")

# User selects model for visualization
model_choice = st.selectbox("Select Model for Visualization", ["Random Forest", "Decision Tree", "Linear Regression"])

# Generate data for visualization
time_range = np.linspace(0, 200, 500)
frequency = input_df['Frequency(Kmph)'][0]
weight = input_df['Weight(Kg)'][0]

# Prediction function for selected model
def predict_critical_temperature(model, time, frequency, weight):
    return model.predict([[time, frequency, weight]])[0]

# Select model and generate predictions
if model_choice == "Random Forest":
    critical_temps = [predict_critical_temperature(regressor_rf, t, frequency, weight) for t in time_range]
elif model_choice == "Decision Tree":
    critical_temps = [predict_critical_temperature(regressor_dt, t, frequency, weight) for t in time_range]
else:
    critical_temps = [predict_critical_temperature(regressor_lr, t, frequency, weight) for t in time_range]

# Plot the predicted critical temperatures over time
fig, ax = plt.subplots()
ax.plot(time_range, critical_temps, label=f'Predicted Critical Temperature ({model_choice})')
ax.axvline(x=input_df['time'][0], color='red', linestyle='--', label='Current Time')
ax.set_xlabel('Time')
ax.set_ylabel('Predicted Critical Temperature')
ax.legend()
ax.set_title(f'Critical Temperature Prediction over Time ({model_choice})')

st.pyplot(fig)
