import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("""
Predicting the chance of having diabetes based on some attributes

""")

pregnancy = st.slider("Have you experienced Preganancy? If yes, please mention the times.",1,12)
pregnancy = int(pregnancy)

glucose = st.slider("Enter current glucose level",44.0,200.0)
glucose = float(glucose)

bp= st.slider("Enter Blood Pressure Level",50.0,120.0)
bp = float(glucose)

skin_thickness = st.slider("Enter Skin Thickness",15,50)
skin_thickness = int(skin_thickness)

##insulin 0 1 2

insulin = st.selectbox("Select Insulin Level",("Low","Medium","High"))

if insulin == "High":
    insulin = 2
elif insulin == "Medium":
    insulin = 1
elif insulin =="Low":
    insulin =0

bmi= st.slider("Enter BMI",5.0,50.0)
bmi = float(bmi)

dpf= st.selectbox("Select Diabetes Pedigree Function Level",("Low","Medium","High"))

if dpf == "High":
    dpf = 2
elif dpf == "Medium":
    dpf = 1
elif dpf =="Low":
    dpf =0
age= st.slider("Enter Age",20,65)
age = int(age)

features = pd.DataFrame([pregnancy,glucose,bp,skin_thickness,insulin,bmi,dpf,age]).values

features = features.reshape(1,-1)
st.write(features)

model = joblib.load("../model.bin")

prediction = model.predict_proba(features)
prediction = prediction[:,1]

st.write(f"Probability of having diabetes:{prediction}")

st.write("Creator : @Raman-rd")


