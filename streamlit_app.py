import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('BrakingEvents.csv')
    return data

data = load_data()

# Preprocess the data
data['Frequency(Kmph)'] = data['Frequency(Kmph)'].astype(float)
data['Weight(Kg)'] = data['Weight(Kg)'].astype(float)

X = data[['time', 'Frequency(Kmph)', 'Weight(Kg)']]
y = data['critical_temp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return rf, dt, lr

regressor_rf, regressor_dt, regressor_lr = train_models(X_train, y_train)

# Title and description
st.title("Brake Application Prediction")
st.write("""
This app predicts the critical temperature for applying brakes based on time, frequency, and weight.
""")

# Create input fields for user input
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

# Prediction using Random Forest model
rf_prediction = regressor_rf.predict(input_df)[0]

# Prediction using Decision Tree model
dt_prediction = regressor_dt.predict(input_df)[0]

# Prediction using Linear Regression model
lr_prediction = regressor_lr.predict(input_df)[0]

# Display predictions
st.subheader("Predicted Critical Temperature for Applying Brakes")
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

if model_choice == "Random Forest":
    critical_temps = [regressor_rf.predict([[t, frequency, weight]])[0] for t in time_range]
elif model_choice == "Decision Tree":
    critical_temps = [regressor_dt.predict([[t, frequency, weight]])[0] for t in time_range]
else:
    critical_temps = [regressor_lr.predict([[t, frequency, weight]])[0] for t in time_range]

# Plot the predicted critical temperatures over time
fig, ax = plt.subplots()
ax.plot(time_range, critical_temps, label=f'Predicted Critical Temperature ({model_choice})')
ax.axvline(x=input_df['time'][0], color='red', linestyle='--', label='Current Time')
ax.set_xlabel('Time')
ax.set_ylabel('Predicted Critical Temperature')
ax.legend()
ax.set_title(f'Critical Temperature Prediction over Time ({model_choice})')

st.pyplot(fig)
