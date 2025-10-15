import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('model', exist_ok=True)

# Load data
df = pd.read_csv('data/raw/data.csv')
df = df.dropna()

print("📊 Original Data:")
print(f"Total rows: {len(df)}")

# Check correlation
print("\n🔍 Correlation Analysis:")
corr_matrix = df[['age', 'years_exp', 'salary']].corr()
print(corr_matrix)
print(f"\nAge vs Years_exp correlation: {corr_matrix.loc['age', 'years_exp']:.3f}")

# SOLUSI 1: HAPUS AGE (karena years_exp lebih penting untuk salary)
print("\n💡 Using features WITHOUT age to avoid multicollinearity")
X = df[['gender', 'education', 'years_exp']]
y = df['salary']

print(f"\n📈 Features used: {X.columns.tolist()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
print("\n📊 Model Coefficients:")
print(f"  Intercept: ${model.intercept_:,.2f}")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name:12s}: ${coef:,.2f}")

# Scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\n✅ Training R²: {train_score:.4f}")
print(f"✅ Test R²: {test_score:.4f}")

# Save
joblib.dump(model, 'model/salary_model.pkl')
print(f"\n💾 Model saved to model/salary_model.pkl")

# Test predictions
print("\n🧪 Test Predictions (Female, Bachelor's):")
for exp in [0, 5, 10, 20]:
    pred = model.predict([[0, 0, exp]])[0]  # gender=0(F), edu=0(Bach), years_exp
    print(f"  {exp:2d} years exp → ${pred:,.0f}")

print("\n✅ Years_exp coefficient should now be POSITIVE!")
