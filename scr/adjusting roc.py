import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
import optuna
import matplotlib.pyplot as plt
import joblib
from yellowbrick.classifier import ROCAUC
import pandas as pd

df = pd.read_csv("input/cleaned_data.csv")

X = df.drop("Outcome",axis=1).values
y = df.Outcome.values

X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.25, random_state=42,stratify=y)

model = ensemble.AdaBoostClassifier(n_estimators=301, learning_rate= 0.028034502100806144)


model.fit(X_train,y_train)

predictions = model.predict_proba(X_test)

df_pred = pd.DataFrame(predictions)


predictions_adj = df_pred.iloc[:,-1]
print(predictions_adj)
predictions_adj = np.where(predictions_adj>0.45,1,0)


print(metrics.confusion_matrix(y_test,predictions_adj))
print(metrics.classification_report(y_test,predictions_adj))