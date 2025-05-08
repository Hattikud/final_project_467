import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from processing import processData

#Read in data
data=pd.read_csv("data/water_potability.csv")
X_train, y_train, X_dev, y_dev, X_test, y_test = processData(data)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
    max_depth=10, random_state=0)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)
y_test_pred = model.predict(X_test)

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
#plt.show()


