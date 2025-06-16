
import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("marketing_campaign1.xlsx")

def main():
    st.title("Customer Segmentation App")

    model = load_model()
    data = load_data()

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Segment Customers")
    if st.button("Predict"):
        segments = model.predict(data.select_dtypes(include=[np.number]))
        data['Segment'] = segments
        st.write("### Segmented Data")
        st.dataframe(data)

if __name__ == '__main__':
    main()

