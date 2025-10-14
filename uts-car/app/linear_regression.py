from app.linear_regression_model import LinearRegressionModel
from app.data_processor import DataProcessor

dp = DataProcessor('data/raw/salary.csv')
X, y, feature_names = dp.process()

model = LinearRegressionModel(use_sklearn=False)
model.fit(X, y, feature_names)
