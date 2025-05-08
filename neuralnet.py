import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from processing import processData

#Read in data
data=pd.read_csv("data/water_potability.csv")
X_train, y_train, X_dev, y_dev, X_test, y_test = processData(data)

#Create MLPClassifier using lbfgs
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_dev_pred = clf.predict(X_dev)
y_test_pred = clf.predict(X_test)

from sklearn import metrics
#Accuracy Scores
print("Training Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
print("Dev Accuracy:",metrics.accuracy_score(y_dev, y_dev_pred))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))