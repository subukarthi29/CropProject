# 🌾 Smart Agriculture System - Crop Recommendation & Yield Prediction

This project presents a smart agriculture solution that integrates **Machine Learning**, **Data Clustering**, and **Visualization** for better **crop planning** and **yield forecasting**. It includes models for crop yield prediction, clustering of crop suitability, and a Streamlit-based interactive interface.

## 📂 Project Structure

📁 CropProject/
├── crop_yield.py # Predicts crop yield using PCA and Random Forest
├── crop-cluster.py # Clusters crops using KMeans based on soil/weather features
├── app.py # Streamlit frontend integrating the prediction and clustering
├── requirements.txt # Python dependencies


## 🚀 Features

- 📊 **Crop Yield Prediction** using PCA and Random Forest Regressor.
- 🌱 **Crop Clustering** with KMeans based on nutrients, pH, temperature, and rainfall.
- 🧠 **Model Training** and storage using Joblib for real-time predictions.
- 🌐 **Interactive Dashboard** with Streamlit for farmers/agronomists.
- 📈 **Visualizations**: Heatmaps, PCA plots, Actual vs Predicted graphs.

## 📌 Tech Stack

- **Python** (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly)
- **Streamlit** for interactive web UI
- **Jupyter Notebook** (for initial experimentation)
- **ESP32 & Sensors** *(optional, for real-time data input)*
- **Joblib** for model persistence

## 📁 Datasets Used

- `crop_yield.csv` – Contains features like Area, Production, Rainfall, Fertilizer, etc. used for yield prediction.
- `Crop_Recommendation.csv` – Contains nutrient and environmental data for clustering.

## 🔍 How to Run

1. clone the repository 
   ```bash
   git clone https://github.com/subukarthi29/CropProject.git
   cd CropProject
2   Install dependencies

pip install -r requirements.txt

3.Run the Streamlit App

streamlit run app.py
