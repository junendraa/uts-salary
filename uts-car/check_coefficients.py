import joblib
import numpy as np

# Load model
model = joblib.load('model/salary_model.pkl')

print("=" * 60)
print("🔍 MODEL ANALYSIS")
print("=" * 60)

print("\n📊 Model Coefficients:")
print(f"  Intercept (base salary): ${model.intercept_:,.2f}")
print(f"\nFeature impacts (per unit increase):")
print(f"  [0] Age:        ${model.coef_[0]:,.2f}")
print(f"  [1] Gender:     ${model.coef_[1]:,.2f}")
print(f"  [2] Education:  ${model.coef_[2]:,.2f}")
print(f"  [3] Years_exp:  ${model.coef_[3]:,.2f}")

print("\n" + "=" * 60)
print("🧪 TEST PREDICTIONS")
print("=" * 60)

# Test: Same person, different experience
print("\n👤 Female, 34 years old, Bachelor's degree:")
test_cases = [
    (34, 0, 0, 2),   # 2 years exp
    (34, 0, 0, 4),   # 4 years exp
    (34, 0, 0, 10),  # 10 years exp
]

for age, gender, edu, exp in test_cases:
    features = np.array([[age, gender, edu, exp]])
    pred = model.predict(features)[0]
    print(f"  {exp:2d} years exp → ${pred:,.2f}")

print("\n💡 If salary DECREASES with more experience, coefficient [3] is wrong!")
