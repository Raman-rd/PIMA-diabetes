import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df):

    df["Pregnancies"] = np.where(df["Pregnancies"]>12,12,df["Pregnancies"])

    df["Glucose"] = df["Glucose"].replace(0,df["Glucose"].median())

    df.BloodPressure = df["BloodPressure"].replace(0,df["BloodPressure"].mean())
    ## Detecting outliers in BP
    IQR = df["BloodPressure"].quantile(0.75) - df["BloodPressure"].quantile(0.25)
    lower_bound=df["BloodPressure"].quantile(0.25)-1.5*IQR
    upper_bound = df["BloodPressure"].quantile(0.75)+1.5*IQR
    df.BloodPressure = np.where(df["BloodPressure"]<lower_bound,lower_bound,df["BloodPressure"])
    df.BloodPressure = np.where(df["BloodPressure"]>upper_bound,upper_bound,df["BloodPressure"])

    ## Fixing Skin Thickness

    df.SkinThickness=df["SkinThickness"].replace(0,np.nan).ffill()

    def bin_insulin(x):
        if x>=0 and x <30:
         return "Normal_Insulin"
        elif x >=30 and x < 127:
         return "Medium_Insulin"
        elif x >=127:
         return "High_Insulin"

    df["Insulin"] = df["Insulin"].apply(bin_insulin)

    # dummies_insulin = pd.get_dummies(df["Insulin"],drop_first=True)
    # df = pd.concat([df,dummies_insulin],axis=1)
    # df.drop("Insulin",axis=1,inplace=True)
    encoder = LabelEncoder()
    df["Insulin"] = encoder.fit_transform(df["Insulin"])

    df.BMI = np.where(df.BMI>50,50,df.BMI)

    def bin_pedigree(x):
        if x >=0.07 and x <0.24:
            return "Low_Pedigree"
        elif x >=0.24 and x < 0.37:
            return "Normal_Pedigree"
        elif x >= 0.37:
             return "High_Pedigree"

    df["DiabetesPedigreeFunction"]=df["DiabetesPedigreeFunction"].apply(bin_pedigree)

    # dummies_pedigree= pd.get_dummies(df["DiabetesPedigreeFunction"],drop_first=True)
    # df = pd.concat([df,dummies_pedigree],axis=1)
    # df.drop("DiabetesPedigreeFunction",axis=1,inplace=True)
    encoder = LabelEncoder()
    df["DiabetesPedigreeFunction"] = encoder.fit_transform(df["DiabetesPedigreeFunction"])


    
    df.to_csv("input/cleaned_data.csv",index=False)


if __name__ == "__main__":

    df = pd.read_csv("input/diabetes.csv")

    clean_data(df)

