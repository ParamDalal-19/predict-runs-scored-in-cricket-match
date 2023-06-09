import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("final_dataset.csv")

# Define X and Y variables
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]].values
Y = df.iloc[:, [9]].values

splits = [[0.1,7], [0.2,7], [0.3,7], [0.4,8]]
for pair in splits:
    # Split the dataset 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=pair[0], random_state=42)

    # Define the Lasso regression model
    lasso_model = Lasso(alpha=0.1)

    lasso_model.fit(X_train, Y_train)

    print(lasso_model.coef_)
