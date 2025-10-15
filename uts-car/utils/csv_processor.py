import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

class CSVAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def load_and_validate(self):
        """Load CSV and validate structure"""
        try:
            self.df = pd.read_csv(self.filepath)
            
            # Check if DataFrame is empty
            if self.df.empty:
                return False, "CSV file is empty"
            
            # Check if has at least 2 columns (1 feature + 1 target)
            if len(self.df.columns) < 2:
                return False, "CSV must have at least 2 columns (features and target)"
            
            # Check for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return False, "CSV must have at least 2 numeric columns"
            
            return True, "CSV loaded successfully"
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def prepare_data(self, target_column=None):
        """Prepare data for regression"""
        try:
            # Select only numeric columns
            numeric_df = self.df.select_dtypes(include=[np.number])
            
            # If target not specified, use last column
            if target_column is None:
                target_column = numeric_df.columns[-1]
            
            # Check if target exists
            if target_column not in numeric_df.columns:
                return False, f"Target column '{target_column}' not found"
            
            # Prepare features and target
            X = numeric_df.drop(columns=[target_column])
            y = numeric_df[target_column]
            
            # Remove rows with missing values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return False, "Not enough data points (minimum 10 required)"
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.results['feature_names'] = X.columns.tolist()
            self.results['target_name'] = target_column
            self.results['n_samples'] = len(X)
            self.results['n_features'] = len(X.columns)
            
            return True, "Data prepared successfully"
        except Exception as e:
            return False, f"Error preparing data: {str(e)}"
    
    def train_model(self):
        """Train linear regression model"""
        try:
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            self.results['train_r2'] = r2_score(self.y_train, y_train_pred)
            self.results['test_r2'] = r2_score(self.y_test, y_test_pred)
            self.results['train_rmse'] = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            self.results['test_rmse'] = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            self.results['test_mae'] = mean_absolute_error(self.y_test, y_test_pred)
            
            # Get coefficients
            self.results['intercept'] = float(self.model.intercept_)
            self.results['coefficients'] = {
                name: float(coef) 
                for name, coef in zip(self.results['feature_names'], self.model.coef_)
            }
            
            # Store predictions for visualization
            self.results['predictions'] = {
                'train': {
                    'actual': self.y_train.tolist(),
                    'predicted': y_train_pred.tolist()
                },
                'test': {
                    'actual': self.y_test.tolist(),
                    'predicted': y_test_pred.tolist()
                }
            }
            
            return True, "Model trained successfully"
        except Exception as e:
            return False, f"Error training model: {str(e)}"
    
    def get_summary(self):
        """Get complete analysis summary"""
        return self.results
