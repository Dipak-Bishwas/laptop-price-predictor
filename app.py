import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
df = pickle.load(open('model/df.pkl', 'rb'))

st.title("Laptop Predictor")

# Brand, Type, and RAM in one row
col1, col2, col3 = st.columns(3)
with col1:
    company = st.selectbox('Brand', df['Company'].unique())
with col2:
    type = st.selectbox('Type', df['TypeName'].unique())
with col3:
    ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight, Touchscreen, and IPS in one row
col4, col5, col6 = st.columns(3)
with col4:
    weight = st.number_input('Weight of the Laptop')
with col5:
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
with col6:
    ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size, Resolution, and CPU in one row
col7, col8, col9 = st.columns(3)
with col7:
    screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)
with col8:
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160',
                                                     '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
with col9:
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD, SSD, and GPU in one row
col10, col11, col12 = st.columns(3)
with col10:
    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
with col11:
    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
with col12:
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    predicted_price = int(np.exp(pipe.predict(query)[0]))
    
    st.markdown(
        f"<span style='font-weight: bold; font-size: 24px;'>The predicted price of this configuration is </span>"
        f"<span style='color: green; font-weight: bold; font-size: 24px;'>{predicted_price}₹</span>",
        unsafe_allow_html=True
    )
