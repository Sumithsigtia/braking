import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load the model
regressor_rf = joblib.load(open('rf_braking.joblib'),'rb'))

regressor_lr = pickle.load(open('lr_braking.sav', 'rb'))

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

# Prediction using Linear Regression model
lr_prediction = regressor_lr.predict(input_df)[0]

# Display predictions
st.subheader("Predicted Critical Temperature for Applying Brakes")
st.write(f"**Random Forest Prediction:** {rf_prediction:.2f}")
st.write(f"**Linear Regression Prediction:** {lr_prediction:.2f}")

# Visualization section
st.subheader("Visualize Prediction")

# User selects model for visualization
model_choice = st.selectbox("Select Model for Visualization", ["Random Forest", "Linear Regression"])

# Generate data for visualization
time_range = np.linspace(0, 200, 500)
frequency = input_df['Frequency(Kmph)'][0]
weight = input_df['Weight(Kg)'][0]

if model_choice == "Random Forest":
    critical_temps = [regressor_rf.predict([[t, frequency, weight]])[0] for t in time_range]
else:
    critical_temps = [regressor_lr.predict([[t, frequency, weight]])[0] for t in time_range]

# Plot the predicted critical temperatures over time
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(time_range, critical_temps, label=f'Predicted Critical Temperature ({model_choice})')
ax.axvline(x=input_df['time'][0], color='red', linestyle='--', label='Current Time')
ax.set_xlabel('Time')
ax.set_ylabel('Predicted Critical Temperature')
ax.legend()
ax.set_title(f'Critical Temperature Prediction over Time ({model_choice})')

st.pyplot(fig)
