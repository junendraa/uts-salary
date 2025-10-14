import sys
sys.path.insert(0, '/home/junendraa/uts-car')

from app.data_processor import DataProcessor
from app.linear_regression import LinearRegression

# Load data
dp = DataProcessor('data/raw/data.csv')
X, y, feature_names = dp.process()

# Train model
print("\n" + "="*50)
print("TRAINING LINEAR REGRESSION MODEL")
print("="*50)

lr = LinearRegression()
lr.fit(X, y, feature_names=feature_names)

# Save model info
model_info = lr.get_model_info()
print(f"\nModel Info: {model_info}")
print("\n✓ Training complete!")
