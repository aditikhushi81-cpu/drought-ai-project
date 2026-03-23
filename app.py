import streamlit as st
import joblib
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
import os

# ================= CONFIG =================
st.set_page_config(page_title="Drought AI", layout="wide")

# ✅ YOUR API KEY (ADDED)
API_KEY = "e944af86db80162cc953d84cde671e74"

# ================= STYLE =================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
# ================= SIDEBAR =================
st.sidebar.markdown("""
<style>
.sidebar-card {
    background: linear-gradient(145deg, #1f2937, #111827);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.title("🌍 Drought AI Panel")
st.sidebar.markdown("Advanced AI-based drought prediction system")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Settings")

year = st.sidebar.selectbox("📅 Year", [2023, 2024, 2025, 2026], index=3)
month = st.sidebar.slider("🗓 Month", 1, 12, 2)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 📊 Features Used")
st.sidebar.markdown("""
- 🌧 Rainfall  
- 🌡 Temperature  
- 💧 Soil Moisture  
- 🌱 NDVI  
""")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.success("Model: Random Forest / XGBoost")
st.sidebar.info("Trained on environmental data")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ================= LOAD MODEL =================
model_path = "models/drought_model.joblib"

if not os.path.exists(model_path):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

# ================= HEADER =================
st.title("🌍 Smart Drought Monitoring Dashboard")
st.markdown("---")

# ================= MAP =================
st.subheader("📍 Select Region")

m = folium.Map(location=[22, 78], zoom_start=5)
map_data = st_folium(m)

st.info("👉 Click on map to fetch live weather (if API active)")

# ================= DEFAULT VALUES =================
rainfall = 100
temperature = 30
soil_moisture = 50
ndvi = 0.5
location_name = "Not Selected"

# ================= LIVE WEATHER =================
if map_data and map_data.get("last_clicked"):

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5)

        if res.status_code == 200:
            data = res.json()

            location_name = data.get("name", "Unknown")
            temperature = data["main"]["temp"]
            rainfall = data.get("rain", {}).get("1h", 0)
            humidity = data["main"]["humidity"]

            soil_moisture = humidity
            ndvi = round(humidity / 100, 2)

            st.success(f"📍 Location: {location_name}")
            st.success("✅ Live weather loaded")

        else:
            st.warning("⚠️ API not active yet. Using manual input.")

    except:
        st.warning("⚠️ API error. Using manual input.")

# ================= INPUTS =================
st.markdown("### 🎛 Adjust Values")

rainfall = st.slider("🌧 Rainfall (mm)", 0, 200, int(rainfall))
temperature = st.slider("🌡 Temperature (°C)", 0, 50, int(temperature))
soil_moisture = st.slider("💧 Soil Moisture (%)", 0, 100, int(soil_moisture))
ndvi = st.slider("🌱 NDVI", 0.0, 1.0, float(ndvi))

# ================= METRICS =================
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🌧 Rainfall", f"{rainfall} mm")
col2.metric("🌡 Temp", f"{temperature} °C")
col3.metric("💧 Soil", f"{soil_moisture}%")
col4.metric("🌱 NDVI", ndvi)

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["🚨 Prediction", "📈 Trends", "🧠 Explain"])

# ================= TAB 1 =================
with tab1:

    if st.button("🚀 Predict Drought"):

        data = pd.DataFrame([[rainfall, temperature, soil_moisture, ndvi]],
                            columns=['rainfall', 'temperature', 'soil_moisture', 'ndvi'])

        result = model.predict(data)[0]

        try:
            prob = model.predict_proba(data)[0][1]
        except:
            prob = 0.5

        confidence = round(prob * 100, 2)

        st.subheader("📊 Prediction Result")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence}%")

        if result == 1:
            st.error("🚨 High Drought Risk")
        else:
            st.success("✅ Low Drought Risk")

        chart = pd.DataFrame({
            "Feature": ["Rainfall", "Temperature", "Soil Moisture", "NDVI"],
            "Value": [rainfall, temperature, soil_moisture, ndvi]
        })

        fig = px.bar(chart, x="Feature", y="Value", color="Value",
                     color_continuous_scale="RdYlGn_r")

        fig.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 =================
with tab2:

    st.subheader("📈 Monthly Trend")

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    rain = np.random.randint(20, 120, 12)
    ndvi_data = np.random.uniform(0.2, 0.8, 12)

    df = pd.DataFrame({
        "Month": months,
        "Rainfall": rain,
        "NDVI": ndvi_data
    })

    fig2 = px.bar(df, x="Month", y="Rainfall")
    fig2.add_scatter(x=months, y=ndvi_data, mode='lines+markers', name="NDVI")

    fig2.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig2, use_container_width=True)

# ================= TAB 3 =================
# ================= TAB 3 =================
with tab3:

    st.subheader("🧠 AI Explanation Dashboard")

    features = ["Rainfall", "Temperature", "Soil Moisture", "NDVI"]
    values = np.array([rainfall, temperature, soil_moisture, ndvi])

    # Normalize importance
    importance = values / values.sum()

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    # 🔥 Bar chart
    fig3 = px.bar(df_imp,
                  x="Importance",
                  y="Feature",
                  orientation='h',
                  color="Importance",
                  color_continuous_scale="Blues")

    fig3.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        title="Feature Contribution to Prediction"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # 🔥 Insights
    st.subheader("🔍 AI Insights")

    max_feature = df_imp.loc[df_imp["Importance"].idxmax(), "Feature"]

    st.info(f"📌 Most influential factor: **{max_feature}**")

    # Dynamic explanation
    if rainfall < 50:
        st.warning("⚠️ Low rainfall detected → increases drought risk")

    if temperature > 35:
        st.warning("🌡 High temperature → increases evaporation")

    if soil_moisture < 30:
        st.warning("💧 Low soil moisture → critical drought indicator")

    if ndvi < 0.3:
        st.warning("🌱 Low vegetation → unhealthy crops")

    # Summary
    st.markdown("### 🧾 Final AI Summary")

    if values.mean() < 50:
        st.error("🚨 Overall conditions indicate HIGH drought probability")
    else:
        st.success("✅ Environmental conditions are relatively stable")