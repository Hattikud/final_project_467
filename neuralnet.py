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
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10, 2), random_state=1, max_iter=1000)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_dev_pred = clf.predict(X_dev)
y_test_pred = clf.predict(X_test)

from sklearn import metrics
#Accuracy Scores
print("Training Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
print("Dev Accuracy:",metrics.accuracy_score(y_dev, y_dev_pred))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))

#Heatmap
cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()