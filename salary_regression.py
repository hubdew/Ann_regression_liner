import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np

# Load the trained model
##model = tf.keras.models.load_model('model.h5')
##model = tf.keras.models.load_model('/Users/admin/Documents/Deep_learn/ven_deep/regression_model.h5')
model = tf.keras.models.load_model('regression_model.h5')

import os
print("Current working directory:", os.getcwd())

# Load the encoder and scaler
with open('onehot_enco_geo.pkl', 'rb') as file:
    onehot_enco_geo = pickle.load(file)

with open('label_encod_gender.pkl', 'rb') as file:
    label_encod_gender = pickle.load(file)

with open('scaler1.pkl', 'rb') as file:
    scaler1 = pickle.load(file)


# Streamlit app
st.title('Estimated salary')

# User input
geography = st.selectbox('Geography', onehot_enco_geo.categories_[0])
gender = st.selectbox('Gender', label_encod_gender.classes_)
age = st.slider('Age', 18, 95)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
##estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encod_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
       
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_enco_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_enco_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler1.transform(input_data)
    print(input_data_scaled.shape)
   ## print(X_train.shape)  # or whatever variable you used for training input


    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    ##st.write(f'Predicted Estimated Salary: {prediction_result:.2f}')
    st.write(f'Predicted Estimated Salary: {prediction_proba:.2f}')

