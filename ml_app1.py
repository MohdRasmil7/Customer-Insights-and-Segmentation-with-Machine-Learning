# -*- coding: utf-8 -*-
"""
Created on Wed May 22 07:26:32 2024

@author: Muhammed Rasmil
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn import decomposition,preprocessing

# Load the trained model (make sure to provide the correct path)
pickle_in = open("final_model.pkl", "rb")
classifier = pickle.load(pickle_in)

# Function to scale features, transform them with PCA and predict using the classifier
def predict_cluster(features):
    # Convert input data into a numpy array
    input_data = np.array(features).reshape(1, -1)
    #n_samples, n_features=20
    # Scale the features
    scaler = preprocessing.StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    

    # Predict using the classifier
    prediction = classifier.predict(input_scaled)
    
    
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
    input_features = [[Education, Marital_Status, Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children]]
    
    # When the 'Predict' button is clicked
    if st.button("Predict"):
        result = predict_cluster(input_features)
        st.success(f'The predicted cluster is: {result}')
    
    # About section
    if st.button("About"):
        st.text("This is a Streamlit app that classifies data into clusters.")

if __name__ == '__main__':
    main()
    

