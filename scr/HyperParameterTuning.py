import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
import optuna

df = pd.read_csv("input/cleaned_data.csv")

X = df.drop("Outcome",axis=1).values
y = df.Outcome.values

X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.25, random_state=42,stratify=y)


def objective(trial):

    n_estimators = trial.suggest_int("n_esimators",7,800)
    learning_rate = trial.suggest_float("learning_rate",0.00001,0.1)

    clf = ensemble.AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)

    return model_selection.cross_val_score(clf,X,y,cv=5,n_jobs=-1).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

trial = study.best_trial

print(f"Acc {trial.value}")
print(f"Best hyper {trial.params}")