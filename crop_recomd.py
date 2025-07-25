import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("Crop_Recommendation.csv")

# Define numerical and target columns
numerical_data = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
x = df[numerical_data]
y = df['Crop']

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)  # Scaling 7 features

# Apply PCA
pca = PCA(n_components=5)  # Reduce dimensions to 5
X_pca = pca.fit_transform(X_scaled)

# Convert PCA results to DataFrame (optional for visualization)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print("PCA Transformed Data:\n", pca_df.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)  # ✅ FIXED ERROR

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Save trained models & scaler
joblib.dump(rf_model, "rf_recommendation.pkl")  # Save trained Random Forest model
joblib.dump(pca, "pca_recommendation.pkl")      # Save PCA transformation
joblib.dump(scaler, "scaler_recommendation.pkl")  # ✅ FIXED: Save scaler instead of X_scaled

print(f"Before PCA, StandardScaler Input Shape: {X_scaled.shape}")  # Should print (num_samples, 7)
