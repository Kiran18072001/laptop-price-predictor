# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:14:29 2021

@author: ajaya
"""

import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('Pipeline.pkl', 'rb'))
df = pickle.load(open('DataFrame.pkl', 'rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# ram of laptop
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the laptop')

# touchscreen
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

# ips
ips = st.selectbox('Ips', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# screen Resolution
res = st.selectbox('Screen Resolution', [
                   '1920x1080', '1366x788', '1600x900',
                   '3840x2160', '3200x1800', '2800x1800', '2560x1440', '2304x1440'])

#cpu
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

#hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict the Price:'):

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(res.split('x')[0])
    Y_res = int(res.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query.reshape(1, 12)
    st.title(int(np.exp(pipe.predict(query)[0])))