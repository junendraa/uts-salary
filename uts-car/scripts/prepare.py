import pandas as pd
import numpy as np

# Load data original
df = pd.read_csv('data/raw/data.csv')

print(f"Original data: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Create numeric version
df_numeric = pd.DataFrame()

# Features
df_numeric['age'] = df['Age'].astype(float)
df_numeric['years_exp'] = df['Years of Experience'].astype(float)

# Encode Gender: Male=1, Female=0
df_numeric['gender'] = (df['Gender'] == 'Male').astype(int)

# Encode Education: Bachelor's=0, Master's=1, PhD=2
education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
df_numeric['education'] = df['Education Level'].map(education_map)

# Target: Salary
df_numeric['salary'] = df['Salary'].astype(float)

# Remove any NaN
df_numeric = df_numeric.dropna()

# Save ke data/raw/data.csv
df_numeric.to_csv('data/raw/data.csv', index=False)

print(f"\n✓ Data converted: {len(df_numeric)} rows")
print(f"\nFirst 5 rows:")
print(df_numeric.head())
print(f"\nData summary:")
print(df_numeric.describe())
