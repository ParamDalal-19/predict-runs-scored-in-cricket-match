import inline as inline
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("final_dataset.csv")

# Define X and Y variables
X = df.iloc[:, [2,3,7,8,10,11]].values
Y = df.iloc[:, [9]].values

splits = [[0.1,7], [0.2,7], [0.3,7], [0.4,8]]
for pair in splits:
    # Split the dataset 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=pair[0], random_state=42)

    knn = KNeighborsRegressor(n_neighbors=pair[1])
    # Fit the model 
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred)
    accuracy = 1 - mse / np.var(Y_test)
    r2 = r2_score(Y_test, Y_pred)

    flat_Y_test = [item for sublist in Y_test for item in sublist]
    flat_Y_pred = [item for sublist in Y_pred for item in sublist]

    print("FOR K =", pair[1])
    print(f"Testing data split: {pair[0] * 100}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")


    colors = []
    for i in range(len(flat_Y_test)):
        if flat_Y_test[i] + 5 < Y_pred[i]:
            colors.append('red')
        elif flat_Y_test[i] - 5 > Y_pred[i]:
            colors.append('red')
        else:
            colors.append('blue')
    plt.scatter(Y_test, Y_pred,s=2,c=colors)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("KNNRegressor: Actual vs Predicted Values")
    plt.show()

