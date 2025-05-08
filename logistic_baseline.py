import pandas as pd

data=pd.read_csv("data/water_potability.csv")

#Fill in empty entries with feature average
data = data.fillna(data.mean())

#Only use four features
X=data.loc[:, ["ph", "Hardness", "Solids", "Sulfate"]]
y=data.iloc[:, 9]
#print(X)

#Split into test/training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression

#Create classifier
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
#Fit classifier to training data
logreg.fit(X_train,y_train)
#Make predictions
y_train_pred=logreg.predict(X_train)
y_test_pred=logreg.predict(X_test)

#Accuracy
from sklearn import metrics

print("Training Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))

cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
cnf_matrix

# Libraries used for plots and arrays.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
