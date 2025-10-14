import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Create model directory if not exists
if not os.path.exists('model'):
    os.makedirs('model')

# Load data from CSV (sudah di-prepare oleh scripts/prepare.py)
df = pd.read_csv('data/raw/data.csv')

print("📊 Dataset Info:")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\n🔍 First 5 rows:")
print(df.head())

# Check for missing values
print("\n🔍 Missing values:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

print(f"\n📊 Data after cleaning: {len(df)} rows")

# Prepare features and target
# Data sudah ter-encode dari scripts/prepare.py
X = df[['age', 'gender', 'education', 'years_exp']]
y = df['salary']

print("\n📈 Features used:")
print(X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Save model
joblib.dump(model, 'model/salary_model.pkl')

print("\n✅ Model berhasil di-training dan disimpan!")
print(f"📊 Training score: {train_score:.4f}")
print(f"📊 Test score: {test_score:.4f}")
print(f"\n💾 File saved: model/salary_model.pkl")

# Show some predictions
print("\n🎯 Sample predictions:")
sample = X_test.head(3)
predictions = model.predict(sample)
actuals = y_test.head(3).values

for i, (pred, actual) in enumerate(zip(predictions, actuals)):
    print(f"   Sample {i+1}: Predicted=${pred:,.0f}, Actual=${actual:,.0f}, Diff=${abs(pred-actual):,.0f}")
