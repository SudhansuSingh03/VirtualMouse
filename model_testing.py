import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

def predict_gesture(model, landmarks):
    """Predict gesture based on landmarks using the trained model."""
    landmarks = np.array(landmarks).reshape(-1, 5, 3, 1)  # Reshape for CNN input (5 landmarks, x, y, z)
    prediction = model.predict(landmarks)
    return np.argmax(prediction)


if __name__ == "__main__":
    # Load preprocessed test data
    data = np.load('preprocessed_data.npz')
    X_test, y_test = data['X_test'], data['y_test']

    # Load the trained model
    model = load_model('gesture_recognition_model8.keras')

    # Initialize lists to store predictions and true labels
    y_pred = []
    y_true = []

    # Iterate over the test set to predict multiple gestures
    for i in range(len(X_test)):
        test_gesture = X_test[i].reshape(-1, 5, 3, 1)  # Reshape the gesture for model input
        predicted_label_index = predict_gesture(model, test_gesture)  # Predict the gesture
        y_pred.append(predicted_label_index)
        y_true.append(y_test[i])

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.2f}")

    # Print confusion matrix for more insight
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title("Confusion Matrix of Gesture Prediction")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Print classification report for precision, recall, and F1 score
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Calculate additional metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"\nF1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Plot a bar graph for metrics
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [accuracy, f1, precision, recall]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['skyblue', 'orange', 'lightgreen', 'salmon'])
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.show()
