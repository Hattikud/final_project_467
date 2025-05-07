import pandas as pd
import numpy as np

#Read in data
data=pd.read_csv("data/water_potability.csv")

#Fill in blanks with feature means
data = data.fillna(data.mean())
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

#X=data.loc[:, ["ph", "Hardness", "Solids", "Sulfate"]]
#X=data.iloc[:, [0,9]]
#y=data.iloc[:, 9]

firstDiv = int(0.5 * len(data))
secondDiv = int(0.75 * len(data))
train_data = data[1:firstDiv]
dev_data = data[firstDiv:secondDiv]
test_data = data[secondDiv:]

X_train=train_data.iloc[:, 0:9]
y_train=train_data.iloc[:, 9]
#print(y_train.value_counts(normalize=False))
    
X_dev=dev_data.iloc[:, 0:9]
y_dev=dev_data.iloc[:, 9]
#print(y_dev.value_counts(normalize=False))

X_test=test_data.iloc[:, 0:9]
y_test=test_data.iloc[:, 9]
#print(y_test.value_counts(normalize=False))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
#print(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0, n_estimators=1000)

rfc.fit(X_train, y_train)
y_train_pred = rfc.predict(X_train)
y_dev_pred = rfc.predict(X_dev)
y_test_pred = rfc.predict(X_test)

incorrect_indices = np.where(y_dev_pred != y_dev)[0]
print("Incorrectly classified examples (indices):", incorrect_indices)
print("Regular X dev: ", X_dev)
print("Feature values of incorrect examples:\n", X_dev[incorrect_indices])



# Analyzing the results
from sklearn import metrics
print("Training Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
print("Dev Accuracy:",metrics.accuracy_score(y_dev, y_dev_pred))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)

# Libraries used for plots and arrays.
import matplotlib.pyplot as plt
import seaborn as sns

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
