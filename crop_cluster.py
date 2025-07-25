import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import joblib

# Load dataset
df = pd.read_csv("Crop_Recommendation.csv")

# Select features for clustering
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
X = df[features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering (choose 4–6 clusters based on Elbow or manually)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Save model and scaler for use in app.py
joblib.dump(kmeans, "kmeans_cluster.pkl")
joblib.dump(scaler, "scaler_cluster.pkl")

# Cluster summary: all crops in each cluster
crop_distribution = df.groupby('Cluster')['Crop'].value_counts().unstack().fillna(0).astype(int)

# Display crops per cluster
print("\n✅ Crop Distribution per Cluster:")
for cluster in crop_distribution.index:
    crops_in_cluster = crop_distribution.loc[cluster]
    print(f"\nCluster {cluster}:")
    print(", ".join(crops_in_cluster[crops_in_cluster > 0].index.tolist()))

# Optional: Visualize the crop distribution as a heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(crop_distribution, cmap="YlOrBr", linewidths=0.5, annot=False)
plt.title("Crop Distribution per Cluster")
plt.xlabel("Crops")
plt.ylabel("Cluster")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Add centroids for nutrient visualization
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
centroids['Cluster'] = centroids.index

# Visualize clusters using PCA (2D)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot clusters using Plotly
fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['Crop'])
fig.update_layout(title="Crop Suitability Clusters (PCA View)")
fig.show()

# Cluster Feature Averages (heatmap)
cluster_means = df.groupby('Cluster')[features].mean()
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='YlGnBu')
plt.title("Average Feature Values per Cluster")
plt.show()
