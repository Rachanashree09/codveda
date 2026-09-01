import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv("cleaned_iris.csv")
print("First 5 rows:")
print(df.head())
X = df[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
print("\nClustered Data:")
print(df.head())
print("\nCluster Counts:")
print(df["cluster"].value_counts())
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["sepal_length"],
    y=df["petal_length"],
    hue=df["cluster"],
    palette="viridis",
    s=80
)
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("K-Means Clustering of Iris Dataset")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()