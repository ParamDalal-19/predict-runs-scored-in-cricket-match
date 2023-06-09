import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("final_dataset.csv")

# Define X and Y variables
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]].values
Y = df.iloc[:, [9]].values

# Scale the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


print("Explained variance ratio:", pca.explained_variance_ratio_)


print("Principal components:", pca.components_)
