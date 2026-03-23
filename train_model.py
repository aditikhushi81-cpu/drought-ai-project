import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Create folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/drought_data.csv")

# Show dataset preview
print("\n📊 Dataset Preview:")
print(df.head())

# Features (input)
X = df[['rainfall', 'temperature', 'soil_moisture', 'ndvi']]

# Target (output)
y = df['drought']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("\n🚀 Training model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Save model
model_path = "models/drought_model.joblib"
joblib.dump(model, model_path)

print(f"\n💾 Model saved at: {model_path}")
print("🎉 Training completed successfully!")