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


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

model = HistGradientBoostingClassifier(max_bins=255, max_iter=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

