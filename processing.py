import pandas as pd
import numpy as np

def processData(data):
    #Fill in empty entries with feature average
    data = data.fillna(data.mean())
    #Randomize data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    #Divide data into sets
    firstDiv = int(0.5 * len(data))
    secondDiv = int(0.75 * len(data))
    train_data = data[1:firstDiv]
    dev_data = data[firstDiv:secondDiv]
    test_data = data[secondDiv:]

    #Divide each set into X and y
    X_train=train_data.iloc[:, 0:9]
    y_train=train_data.iloc[:, 9]
    #print(y_train.value_counts(normalize=False))
        
    X_dev=dev_data.iloc[:, 0:9]
    y_dev=dev_data.iloc[:, 9]
    #print(y_dev.value_counts(normalize=False))

    X_test=test_data.iloc[:, 0:9]
    y_test=test_data.iloc[:, 9]

    #Normalize Data
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    #print(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_dev, y_dev, X_test, y_test