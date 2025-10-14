import matplotlib.pyplot as plt
import io, base64

def visualize_regression(X, y, y_pred):
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, y_pred, color='red', label='Predicted')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('ascii')
