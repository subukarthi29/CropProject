cd # === Importing Libraries ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load Dataset ===
df = pd.read_csv("crop_yield.csv")

# === Define Columns ===
categorical_cols = ['Crop', 'Season', 'State']
numerical_cols = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# === Encode Categorical Columns ===
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Feature & Target Separation ===
X = df[categorical_cols + numerical_cols]  # Input features
y = df['Yield']                            # Target variable

# === Scale Numerical Features ===
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# === Apply PCA ===
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Create PCA DataFrame (optional for viewing)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# === Train Random Forest Model ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# === Predict & Evaluate ===
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# === Save Models & Preprocessing Objects ===
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(pca, "pca_transform.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# === Visualizations ===

# Feature Importance (of PCs)
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=[f'PC{i+1}' for i in range(len(importances))], y=importances)
plt.xlabel('Principal Components')
plt.ylabel('Feature Importance')
plt.title('Feature Importance of Principal Components')
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Ideal line
plt.tight_layout()
plt.show()

# Residual Distribution
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.tight_layout()
plt.show()
