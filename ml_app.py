import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image



pickle_in = open("final_model.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_cluster(Education, Marital_Status, Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children):
    
   
   
    prediction=classifier.predict([[Education, Marital_Status, Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children]])
    print(prediction)
    return prediction



def main():
    st.title("Cluster classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit cluster classification ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # Create text input fields for each feature
    Education = st.text_input("Education", "Type Here")
    Marital_Status = st.text_input("Marital_Status", "Type Here")
    Income = st.text_input("Income", "Type Here")
    Kidhome = st.text_input("Kidhome", "Type Here")
    Teenhome = st.text_input("Teenhome", "Type Here")
    Recency = st.text_input("Recency", "Type Here")
    NumDealsPurchases = st.text_input("NumDealsPurchases", "Type Here")
    NumWebPurchases = st.text_input("NumWebPurchases", "Type Here")
    NumCatalogPurchases = st.text_input("NumCatalogPurchases", "Type Here")
    NumStorePurchases = st.text_input("NumStorePurchases", "Type Here")
    NumWebVisitsMonth = st.text_input("NumWebVisitsMonth", "Type Here")
    AcceptedCmp3 = st.text_input("AcceptedCmp3", "Type Here")
    AcceptedCmp4 = st.text_input("AcceptedCmp4", "Type Here")
    AcceptedCmp5 = st.text_input("AcceptedCmp5", "Type Here")
    AcceptedCmp1 = st.text_input("AcceptedCmp1", "Type Here")
    AcceptedCmp2 = st.text_input("AcceptedCmp2", "Type Here")
    Complain = st.text_input("Complain", "Type Here")
    Response = st.text_input("Response", "Type Here")
    total_amount_spent = st.text_input("total_amount_spent", "Type Here")
    Children = st.text_input("Children", "Type Here")
    result=""
    if st.button("Predict"):
        result=predict_cluster(Education, Marital_Status, Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
