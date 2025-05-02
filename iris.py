import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv("Iris.csv")
df.drop("Id", axis=1, inplace=True)

# Encode the target labels
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2. Try different values of K
k_values = [1, 3, 5, 7]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nK = {k}")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)

    # 5. Visualize decision boundaries using PCA (2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    knn.fit(X_pca, y)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolor='k')
    plt.title(f"KNN Decision Boundaries (K={k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=le.classes_.tolist(), title="Species")

    plt.show()
