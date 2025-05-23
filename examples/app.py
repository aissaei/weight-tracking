import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.title("Weight Tracking and Analysis App")

# 1. User imports CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 2. User identifies weight and time columns
    st.subheader("Select Columns")
    columns = df.columns.tolist()
    weight_col = st.selectbox("Select the weight column", columns)
    time_col = st.selectbox("Select the time/date column", columns)

    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)
    df[weight_col] = df[weight_col].str.replace('lb', '').astype(float)

    # 3. Average multiple observations per day
    daily_df = df.groupby(df[time_col].dt.date)[weight_col].mean().reset_index()
    daily_df.columns = ['date', 'weight']
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    # Calculate 7-day moving average
    daily_df['7d_ma'] = daily_df['weight'].rolling(window=7).mean()

    # Plotting
    st.subheader("Weight Over Time with 7-Day Moving Average")
    fig, ax = plt.subplots()
    ax.plot(daily_df['date'], daily_df['weight'], label='Daily Avg Weight', alpha=0.5)
    ax.plot(daily_df['date'], daily_df['7d_ma'], label='7-Day Moving Avg', color='orange', linewidth=2)

    # 4. User provides two dates for regression
    st.subheader("Fit Linear Regression Between Dates")
    min_date = daily_df['date'].min()
    max_date = daily_df['date'].max()
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    mask = (daily_df['date'] >= pd.to_datetime(start_date)) & (daily_df['date'] <= pd.to_datetime(end_date))
    reg_df = daily_df.loc[mask].dropna(subset=['7d_ma'])

    if len(reg_df) >= 2:
        # Prepare data for regression
        X = (reg_df['date'] - reg_df['date'].min()).dt.days.values.reshape(-1, 1)
        y = reg_df['7d_ma'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ax.plot(reg_df['date'], y_pred, color='red', label='Regression Line')

        # Regression formula
        slope = model.coef_[0]
        intercept = model.intercept_
        formula = f"y = {slope:.2f}x + {intercept:.2f}"
        ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=10, verticalalignment='top', color='red')

        # 5. Average weight change per week
        days = (reg_df['date'].max() - reg_df['date'].min()).days
        total_change = y[-1] - y[0]
        weeks = days / 7 if days > 0 else 1
        weekly_change = total_change / weeks if weeks != 0 else 0
        ax.text(0.05, 0.88, f"Avg weight change/week: {weekly_change:.2f}", transform=ax.transAxes, fontsize=10, color='blue')
    else:
        st.warning("Not enough data points for regression between selected dates.")

    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend()
    st.pyplot(fig)

    # 6. Instructions for deployment
    st.markdown("""
    ### Deployment Instructions
    - Push this app and a requirements.txt file (with `streamlit`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`) to a public GitHub repo.
    - Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and deploy your app for free.
    """)

else:
    st.info("Please upload a CSV file to begin.")
