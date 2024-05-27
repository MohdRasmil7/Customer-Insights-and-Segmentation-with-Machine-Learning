# -*- coding: utf-8 -*-
"""

@author: Muhammed Rasmil
"""


import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the trained model and transformers (make sure to provide the correct path)
with open("final_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("pca.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

# Function to scale features, transform them with PCA, and predict using the classifier

def predict_cluster(categorical_features, continuous_features):
    # Convert input data into a numpy array
    continuous_features = np.array(continuous_features).reshape(1, -1)
    categorical_features = np.array(categorical_features).reshape(1, -1)

    # Scale the features using the pre-fitted scaler
    scaled_features = scaler.transform(continuous_features)

    # Transform the features using the pre-fitted PCA
    pca_features = pca.transform(scaled_features)

    # Combine PCA features with categorical features
    final_features = np.concatenate((pca_features, categorical_features), axis=1)

    # Predict using the classifier
    prediction = classifier.predict(final_features)

    return prediction
# Main function for the Streamlit app
def main():
    st.title("Cluster Classification App")
    # HTML for styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Cluster Classification ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create text input fields for each feature
    Education = st.text_input("Education")
    Marital_Status = st.text_input("Marital_Status")
    Income = st.text_input("Income")
    Kidhome = st.text_input("Kidhome")
    Teenhome = st.text_input("Teenhome")
    Recency = st.text_input("Recency")
    NumDealsPurchases = st.text_input("NumDealsPurchases")
    NumWebPurchases = st.text_input("NumWebPurchases")
    NumCatalogPurchases = st.text_input("NumCatalogPurchases")
    NumStorePurchases = st.text_input("NumStorePurchases")
    NumWebVisitsMonth = st.text_input("NumWebVisitsMonth")
    AcceptedCmp3 = st.text_input("AcceptedCmp3")
    AcceptedCmp4 = st.text_input("AcceptedCmp4")
    AcceptedCmp5 = st.text_input("AcceptedCmp5")
    AcceptedCmp1 = st.text_input("AcceptedCmp1")
    AcceptedCmp2 = st.text_input("AcceptedCmp2")
    Complain = st.text_input("Complain")
    Response = st.text_input("Response")
    total_amount_spent = st.text_input("total_amount_spent")
    Children = st.text_input("Children")

    # Creating a list of lists for a single sample
    continuous_features = [[Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children]]
    categorical_features = [[Education, Marital_Status]]

    # When the 'Predict' button is clicked
    if st.button("Predict"):
        result = predict_cluster(categorical_features, continuous_features)
        st.success(f'The predicted cluster is: {result[0]}')

    # About section
    if st.button("About"):
        st.text("""About
Welcome to the Streamlit Cluster Classification ML App! This application leverages a pre-trained machine learning model to classify customer data into meaningful clusters. Whether you’re exploring customer behavior, predicting responses, or segmenting your audience, this app provides insights at your fingertips.

Key Features:

Predictive Power: Utilize the power of machine learning to make informed decisions.
User-Friendly Interface: Input customer attributes, click “Predict,” and discover the predicted cluster.
Data-Driven Insights: Understand customer segments based on their characteristics.
Explore the app, uncover patterns, and enhance your decision-making process. Happy clustering!

Muhammed Rasmil


© 2024.""")
        
    # Credits section
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <hr>
        <p style="color: grey;">© Muhammed Rasmil</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
