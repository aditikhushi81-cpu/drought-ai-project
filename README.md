# 🌍 Smart Drought Monitoring Dashboard

An AI-powered web application that predicts drought risk using environmental data such as rainfall, temperature, soil moisture, and NDVI.

---

## 🚀 Features

* 📍 Interactive map for region selection
* 🌧 Rainfall, 🌡 temperature, 💧 soil moisture, 🌱 NDVI inputs
* 🤖 Machine Learning-based drought prediction
* 📊 Visualization using Plotly
* 🧠 AI explanation of results
* 🌐 Live weather integration (OpenWeather API)

---

## 🛠️ Tech Stack

* Python 3.10
* Streamlit
* Scikit-learn / XGBoost
* Pandas, NumPy
* Plotly
* Folium (maps)

---

## 📂 Project Structure

```
drought-ai-project/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── train_model.py
│
├── models/
│   └── drought_model.joblib
│
├── data/
│   └── drought_data.csv
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
py -3.10 -m streamlit run app.py
```

---

## 🌐 Deployment

This project is deployed using Streamlit Cloud.

---

## 👨‍💻 Author

Adity

---

