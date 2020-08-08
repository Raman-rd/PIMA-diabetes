import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
import optuna
import matplotlib.pyplot as plt
import joblib
from yellowbrick.classifier import ROCAUC

df = pd.read_csv("input/cleaned_data.csv")

X = df.drop("Outcome",axis=1).values
y = df.Outcome.values

X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.25, random_state=42,stratify=y)

model = ensemble.AdaBoostClassifier(n_estimators=301, learning_rate= 0.028034502100806144)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

joblib.dump(model,"model.bin")

accuracy = metrics.accuracy_score(prediction,y_test)

print(metrics.confusion_matrix(y_test,prediction))
print(metrics.classification_report(prediction,y_test))


visualizer =ROCAUC(model,classes=[0,1])
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
visualizer.show()
