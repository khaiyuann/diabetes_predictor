# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:55:09 2022

@author: LeongKY
"""

import os
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

#%% Load model
MODEL_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
SCALER_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model', 'scaler.pkl')
ENCODER_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')
diabetes_dict = {1:'diabetic', 0:'not diabetic'}
                               
diabetes_classifier = load_model(MODEL_LOAD_PATH)
diabetes_classifier.summary()


#%% Load scaler and encoder
scaler = pickle.load(open(SCALER_LOAD_PATH, 'rb'))
encoder = pickle.load(open(ENCODER_LOAD_PATH, 'rb'))

#%% Deploy model (testing)
#patient_info = [5, 116, 0, 25.6, 0.201, 30] #enter data without dropped columns
#patient_info = np.expand_dims(patient_info, 0)
#patient_info = scaler.transform(patient_info)
#prediction = diabetes_classifier.predict(patient_info)

#print(np.argmax(prediction))

#print('This patient is ' + diabetes_dict[np.argmax(prediction)])

#%% Streamlit implementation
'''
This streamlit app used the developed model to determine if a patient
is likely to be diabetic or not based on their information.
'''
with st.form('Diabetes Prediction From'):
    st.write('Patient\'s info')
    pregnancies = int(st.number_input('No. of times of pregnancies'))
    glucose = st.number_input('Glucose')
    insulin = st.number_input('Insulin')
    bmi = st.number_input('BMI')
    dpf = st.number_input('Diabetes pedigree function')
    age = int(st.number_input('Age'))
    
    submitted = st.form_submit_button('Submit')
    st.write(submitted)
    
    if submitted == True:
        pat_info = np.array([pregnancies, glucose, insulin, bmi, dpf, age])
        pat_info = np.expand_dims(pat_info,0)
        pat_info = scaler.transform(pat_info)
        prediction = diabetes_classifier.predict(pat_info)
        st.write(np.argmax(prediction))
        st.write('This patient is ' + diabetes_dict[np.argmax(prediction)])