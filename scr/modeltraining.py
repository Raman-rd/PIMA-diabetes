import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
df = pd.read_csv("input/cleaned_data.csv")

X = df.drop("Outcome",axis=1).values
y = df.Outcome.values

# X_scaled = preprocessing.StandardScaler().fit_transform(X)

roc = []
acc=[]

skf = model_selection.StratifiedKFold(n_splits=5)

skf.get_n_splits(X,y)

classifier = ensemble.AdaBoostClassifier()

for train_index, test_index in skf.split(X,y):
    print("Train Index" , train_index,"Validation",test_index)
    X1_train,X1_test = X[train_index],X[test_index]
    y1_train,y1_test = y[train_index],y[test_index]

    classifier.fit(X1_train,y1_train)
    prediction = classifier.predict(X1_test)
    score = metrics.roc_auc_score(prediction,y1_test)
    accuracy = metrics.accuracy_score(prediction,y1_test)
    roc.append(score)
    acc.append(accuracy)
    print(metrics.classification_report(prediction,y1_test))
print(roc)
print(np.mean(acc))
