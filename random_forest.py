def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)


import pandas as pd
# Load the dataset
dataset = pd.read_csv("final_dataset.csv")

# Define X and Y variables
X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].values
Y = dataset.iloc[:, [9]].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Training the dataset 
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, max_features=None)

from sklearn.model_selection import train_test_split
for test_size in [0.1, 0.2, 0.3, 0.4]:
    # Splitting the dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    reg.fit(X_train, y_train.ravel())

    y_pred = reg.predict(X_test)
    score = reg.score(X_test,y_test)*100
    print(f"Test split: {test_size*100}%")
    print("R square value:", score)
    print("Custom accuracy:", custom_accuracy(y_test,y_pred,10))


