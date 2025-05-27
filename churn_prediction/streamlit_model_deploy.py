import pickle
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np




def main():

    Gender = st.text_input("Gender")
    Age = st.number_input("Age")
    Tenure = st.number_input("Tenure")
    Usage_Frequency = st.number_input("Usage Frequency")
    Support_Calls = st.number_input("Support Calls")
    Payment_Delay = st.number_input("Payment Delay")
    Subscription_Type = st.text_input("Subscription Type")
    Contract_Length = st.text_input("Contract Length")
    Total_Spend = st.number_input("Total Spend")
    Last_Interaction = st.number_input("Last Interaction")
    


    data = {"Age": int(Age),
            "Tenure": int(Tenure),
            "Usage Frequency": int(Usage_Frequency), 
            "Support Calls": int(Support_Calls),
            "Payment Delay": int(Payment_Delay),
            "Total Spend": int(Total_Spend),
            "Last Interaction": int(Last_Interaction),
            "Gender": str(Gender),
            "Subscription Type": str(Subscription_Type),
            "Contract Length": str(Contract_Length)
            }
    data = pd.DataFrame([data])


    def make_preprocess(df):
        with open("encode.pkl", "rb") as f:
            encode = pickle.load(f)
        with open("normalize.pkl", "rb") as f:
            normalized = pickle.load(f)

        
        n = pd.DataFrame(normalized.transform(df.select_dtypes("int")))
        e = pd.DataFrame(encode.transform(df.select_dtypes("object")))
        dt_df = pd.concat((n, e), axis=1)
        all_col = np.concatenate((normalized.get_feature_names_out(), encode.get_feature_names_out()))
        dt_df.columns = all_col
        return dt_df
    
    def make_prediction(df):
        prep_df = make_preprocess(df)
        with open("churn_prediction_model.pkl", "rb") as f:
            mp = pickle.load(f)

        return mp.predict(prep_df)
    
    if st.button("Predict"):
        makeprediction = make_prediction(data)
    
    if makeprediction == 1:
        st.success("This Customer will Churn")
    else: st.success("This Customer will Not Churn")
    

if __name__ == "__main__":
    main()


