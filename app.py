import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="smart_agriculture"
    )

def log_yield_prediction(data):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO yield_predictions 
        (crop, season, state, crop_year, area, production, rainfall, fertilizer, pesticide, predicted_yield)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, data)
    conn.commit()
    cursor.close()
    conn.close()

def log_crop_recommendation(data):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO crop_recommendations 
        (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, recommended_crop, predicted_cluster)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, data)
    conn.commit()
    cursor.close()
    conn.close()


# Load models
rf_yield_model = joblib.load("rf_model.pkl")
pca_yield = joblib.load("pca_transform.pkl")
scaler_yield = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

rf_recommendation_model = joblib.load("rf_recommendation.pkl")
pca_recommendation = joblib.load("pca_recommendation.pkl")
scaler_recommendation = joblib.load("scaler_recommendation.pkl")

kmeans_cluster = joblib.load("kmeans_cluster.pkl")
scaler_cluster = joblib.load("scaler_cluster.pkl")

# Use background image from URL
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1591608513281-e5b4f76e0e7e");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .output-box {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            text-align: center;
            font-size: 18px;
            color: #333;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🌾 Smart Agriculture Dashboard")

# Sidebar
option = st.sidebar.radio("Choose Task:", ["Crop Yield Prediction", "Crop Recommendation", "Clustering Analysis"])

# Categorical class lists
crop_classes = ['Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut', 'Cotton(lint)', 'Dry chillies', 'Gram', 'Jute', 'Linseed',
                'Maize', 'Mesta', 'Niger seed', 'Onion', 'Other Rabi pulses', 'Potato', 'Rapeseed &Mustard', 'Rice', 'Sesamum',
                'Small millets', 'Sugarcane', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric', 'Wheat', 'Bajra', 'Black pepper',
                'Cardamom', 'Coriander', 'Garlic', 'Ginger', 'Groundnut', 'Horse-gram', 'Jowar', 'Ragi', 'Cashewnut', 'Banana',
                'Soyabean', 'Barley', 'Khesari', 'Masoor', 'Moong(Green Gram)', 'Other Kharif pulses', 'Safflower', 'Sannhamp',
                'Sunflower', 'Urad', 'Peas & beans (Pulses)', 'other oilseeds', 'Other Cereals', 'Cowpea(Lobia)', 'Oilseeds total',
                'Guar seed', 'Other Summer Pulses', 'Moth']

season_classes = ['Whole Year ', 'Kharif     ', 'Rabi       ', 'Autumn     ','Summer     ', 'Winter     ']
state_classes = ['Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal', 'Puducherry', 'Goa', 'Andhra Pradesh',
                 'Tamil Nadu', 'Odisha', 'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram', 'Punjab',
                 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand',
                 'Jharkhand', 'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana', 'Arunachal Pradesh', 'Sikkim']

# ======================== Crop Yield Prediction ========================
if option == "Crop Yield Prediction":
    st.header("🌱 Crop Yield Prediction")

    crop = st.selectbox("Select Crop:", crop_classes)
    season = st.selectbox("Select Season:", season_classes)
    state = st.selectbox("Select State:", state_classes)
    crop_year = st.number_input("Enter Crop Year:", min_value=2000, max_value=2030, value=2024)
    area = st.number_input("Enter Area (ha):", min_value=0.1, value=1.0)
    production = st.number_input("Enter Production (tons):", min_value=0.1, value=1.0)
    rainfall = st.number_input("Enter Annual Rainfall (mm):", min_value=0.0, value=1000.0)
    fertilizer = st.number_input("Enter Fertilizer Used (kg/ha):", min_value=0.0, value=50.0)
    pesticide = st.number_input("Enter Pesticide Used (kg/ha):", min_value=0.0, value=10.0)

    if st.button("Predict Yield"):
        try:
            crop_encoded = label_encoders['Crop'].transform([crop])[0]
            season_encoded = label_encoders['Season'].transform([season])[0]
            state_encoded = label_encoders['State'].transform([state])[0]
            features = np.array([[crop_encoded, season_encoded, state_encoded, crop_year, area, production, rainfall, fertilizer, pesticide]])
            features[:, 3:] = scaler_yield.transform(features[:, 3:])
            features_pca = pca_yield.transform(features)
            predicted_yield = rf_yield_model.predict(features_pca)[0]
            st.markdown(f'<div class="output-box">🌾 Predicted Yield: {predicted_yield:.2f} tons/ha</div>', unsafe_allow_html=True)
            log_yield_prediction((
                str(crop), str(season), str(state),
                int(crop_year), float(area), float(production),
                float(rainfall), float(fertilizer), float(pesticide),
                float(predicted_yield)
            ))

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ======================== Crop Recommendation + Cluster ========================
elif option == "Crop Recommendation":
    st.header("🌿 Crop Recommendation System + Cluster Prediction")

    nitrogen = st.number_input("Nitrogen:", min_value=0.0, value=50.0)
    phosphorus = st.number_input("Phosphorus:", min_value=0.0, value=50.0)
    potassium = st.number_input("Potassium:", min_value=0.0, value=50.0)
    temperature = st.number_input("Temperature (°C):", min_value=0.0, value=25.0)
    humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH:", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm):", min_value=0.0, value=200.0)

    if st.button("Recommend Crop & Predict Cluster"):
        try:
            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

            # Crop recommendation
            features_scaled = scaler_recommendation.transform(features)
            features_pca = pca_recommendation.transform(features_scaled)
            recommended_crop = rf_recommendation_model.predict(features_pca)[0]

            # Cluster prediction
            cluster_scaled = scaler_cluster.transform(features)
            predicted_cluster = kmeans_cluster.predict(cluster_scaled)[0]

            st.markdown(
                f"""
                <div class="output-box">
                    🌱 Recommended Crop: <strong>{recommended_crop}</strong><br>
                    📍 Belongs to Cluster: <strong>{predicted_cluster}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
            log_crop_recommendation((
                float(nitrogen), float(phosphorus), float(potassium),
                float(temperature), float(humidity), float(ph),
                float(rainfall), str(recommended_crop), int(predicted_cluster)
            ))


        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ======================== Clustering Analysis ========================
elif option == "Clustering Analysis":
    st.header("📊 Crop Clustering Analysis")

    # Load data
    df = pd.read_csv("Crop_Recommendation.csv")
    features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    X = df[features]

    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

    # Display clustering plot
    st.plotly_chart(
        px.scatter(df, x="PCA1", y="PCA2", color=df['Cluster'].astype(str), hover_data=['Crop'])
        .update_layout(title="Cluster View (PCA)")
    )

    # Cluster Feature Averages heatmap
    st.subheader("Cluster Feature Averages (Heatmap)")
    cluster_means = df.groupby('Cluster')[features].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap='YlGnBu')
    plt.title("Average Feature Values per Cluster")
    st.pyplot(plt)
    
    # Display crop summary per cluster
    st.subheader("Cluster Summary (Most Common Crop in Each Cluster)")
    cluster_summary = df.groupby("Cluster")['Crop'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    cluster_summary.columns = ['Cluster', 'Most_Common_Crop']
    st.dataframe(cluster_summary)

    # Crop distribution per cluster (detailed breakdown)
    st.subheader("Detailed Crop Distribution Across Clusters")
    crop_distribution = df.groupby('Cluster')['Crop'].value_counts().unstack().fillna(0).astype(int)

    # Display the crop distribution as a table
    st.dataframe(crop_distribution)

    # Visualize the crop distribution as a vertical heatmap (transpose the data)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(crop_distribution.transpose(), cmap="YlOrBr", annot=False, ax=ax)
    st.pyplot(fig)
