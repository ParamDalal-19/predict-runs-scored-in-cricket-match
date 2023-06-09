import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("final_dataset.csv")

# split the dataset 
X = df.iloc[:, [1, 2, 3, 4, 5, 7, 8, 10, 11]].values
Y = df.iloc[:, [9]].values

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

splits = [0.1, 0.2, 0.3, 0.4]
for split in splits:
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=42)

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = DecisionTreeRegressor()

    # train the model
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    flat_y_test = [item for sublist in y_test for item in sublist]
    accuracy = custom_accuracy(flat_y_test, y_pred,5)

    # print the evaluation metrics
    print(f"Train: {split * 100}%")
    print(f"Overall Accuracy: {accuracy}")

    colors = []
    for i in range(len(flat_y_test)):
        if flat_y_test[i] + 5 < y_pred[i]:
            colors.append('red')
        elif flat_y_test[i] - 5 > y_pred[i]:
            colors.append('red')
        else:
            colors.append('blue')

    plt.scatter(flat_y_test, y_pred, s=2, c=colors)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.show()




