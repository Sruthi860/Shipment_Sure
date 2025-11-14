import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

#Load trained model
model = joblib.load("xgboost_model.pkl")

# App title and description
st.title("ShipmentSure")
st.write("""
This app predicts whether a shipment will be **delivered on time**  
using the trained XGBoost model.
""")

# Sidebar input section
st.sidebar.header("Enter Shipment Details")

# Collect user input
ID = st.sidebar.number_input("Shipment ID", 1, 999999, 1001)
warehouse_block = st.sidebar.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
mode_of_shipment = st.sidebar.selectbox("Mode of Shipment", ['Ship', 'Road', 'Flight'])
customer_care_calls = st.sidebar.number_input("Customer Care Calls", 0, 10, 3)
customer_rating = st.sidebar.slider("Customer Rating", 1, 5, 3)
cost_of_product = st.sidebar.number_input("Cost of the Product", 10, 10000, 300)
prior_purchases = st.sidebar.number_input("Prior Purchases", 0, 20, 3)
product_importance = st.sidebar.selectbox("Product Importance", ['low', 'medium', 'high'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
discount_offered = st.sidebar.number_input("Discount Offered", 0, 80, 10)
weight_in_gms = st.sidebar.number_input("Weight (in grams)", 100, 8000, 2000)

# Derived feature
cost_to_weight_ratio = cost_of_product / weight_in_gms

# Map categorical features to match training preprocessing
def encode_inputs():
    # Label encoding (same as in training)
    product_importance_map = {'low': 1, 'medium': 2, 'high': 0}
    gender_map = {'Male': 1, 'Female': 0}

    # One-hot encoding columns used during training
    warehouse_cols = ['Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F']
    mode_cols = ['Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship']

    # Initialize all features with zero
    data = dict.fromkeys(
        ['ID', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
         'Prior_purchases', 'Product_importance', 'Gender', 'Discount_offered',
         'Weight_in_gms', 'Warehouse_block_B', 'Warehouse_block_C',
         'Warehouse_block_D', 'Warehouse_block_F', 'Mode_of_Shipment_Road',
         'Mode_of_Shipment_Ship', 'Cost_to_Weight_ratio'], 0
    )

    # Fill numeric and encoded values
    data['ID'] = ID
    data['Customer_care_calls'] = customer_care_calls
    data['Customer_rating'] = customer_rating
    data['Cost_of_the_Product'] = cost_of_product
    data['Prior_purchases'] = prior_purchases
    data['Product_importance'] = product_importance_map[product_importance]
    data['Gender'] = gender_map[gender]
    data['Discount_offered'] = discount_offered
    data['Weight_in_gms'] = weight_in_gms
    data['Cost_to_Weight_ratio'] = cost_to_weight_ratio

    # One-hot encode warehouse
    if warehouse_block != 'A':
        data[f'Warehouse_block_{warehouse_block}'] = 1

    # One-hot encode mode of shipment
    if mode_of_shipment == 'Road':
        data['Mode_of_Shipment_Road'] = 1
    elif mode_of_shipment == 'Ship':
        data['Mode_of_Shipment_Ship'] = 1

    return pd.DataFrame([data])

# Prepare final input
input_df = encode_inputs()

# Ensure correct feature order
expected_cols = [
 'ID', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
 'Prior_purchases', 'Product_importance', 'Gender', 'Discount_offered',
 'Weight_in_gms', 'Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D',
 'Warehouse_block_F', 'Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship', 'Cost_to_Weight_ratio'
]


for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_cols]

#  Predict button
if st.button(" Predict Delivery Status"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {'Delivered on Time  (1)' if prediction == 1 else 'Delayed (0)'}")
    st.write(f"**Probability of On-Time Delivery:** {probability:.2f}")

    if prediction == 1 and probability > 0.7:
        st.success(" High confidence: Shipment likely to arrive on time.")
    elif prediction == 0 and probability < 0.3:
        st.error(" High confidence: Shipment likely to be delayed.")
    else:
        st.info(" Medium confidence — prediction uncertain.")


