"""
Visualization utilities for regression analysis
"""

def generate_chart_data(results):
    """
    Generate chart data structure for frontend
    """
    return {
        'scatter': {
            'train': results['predictions']['train'],
            'test': results['predictions']['test']
        },
        'residuals': calculate_residuals(results)
    }

def calculate_residuals(results):
    """
    Calculate residuals for visualization
    """
    test_data = results['predictions']['test']
    residuals = [
        actual - predicted 
        for actual, predicted in zip(test_data['actual'], test_data['predicted'])
    ]
    return {
        'predicted': test_data['predicted'],
        'residuals': residuals
    }
