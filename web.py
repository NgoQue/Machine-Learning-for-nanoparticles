import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.title(" Predict the absorption spectrum of the gold nanosphere given its radius")

st.markdown("""
* Enter the radius R for which you want to predict the absorption spectrum.
* Your plots will appear below
"""  )

number = st.number_input('Please enter the desired size in the box below to generate the corresponding absorption spectrum plots')


def predict_model(R):
    global abs_wl
    extra_model = joblib.load("DecisionRegression preds wl, Q_abs.joblib")

    radius = np.full((1, 200), R)

    abs_wl = extra_model.predict(radius)
    abs_wl = abs_wl.reshape(200, 2)

predict_model(number)

x = abs_wl[ :, 0]
y = abs_wl[ :, 1]


st.write("The absorption spectrum of the gold nanosphere R = ", number)
fig, ax = plt.subplots()
ax.plot(abs_wl[ :, 0], abs_wl[ :, 1])
ax.set_xlabel("wavelength")
ax.set_ylabel("Q_absorption")
st.pyplot(fig)


#-------------------------------------------------#
st.title(" Predict the radius of the gold nanosphere")
st.markdown(""" Please upload the CSV file """)

dataset = st.file_uploader("Upload file here", type = ['csv'])
if dataset is not None:
    data_test = pd.read_csv(dataset)

    data_test = data_test.values

    collumn = data_test.shape[1]
    row = data_test.shape[0]

    for j in np.arange(0, collumn, 1):
        for i in np.arange(1, row, 1):
            if(data_test[i, j] == data_test[i-1, j]+1):
                data_test = np.delete(data_test, j, 1)
        break

    st.write(data_test)

    data_test = data_test.reshape(1, row*2)
    print(data_test)

    def predict_Radius(data):
        global Rad
        KN_model = joblib.load("ExtraRegressor preds R.joblib")
        Rad = KN_model.predict(data)
        
    predict_Radius(data_test)
    st.write("The predicted radius using your data is", Rad[0, 1])


    

