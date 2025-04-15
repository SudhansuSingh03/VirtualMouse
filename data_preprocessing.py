import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_data(csv_file='hand_landmarks1.csv'):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Select the relevant columns for hand landmarks (x, y, z coordinates for 5 landmarks)
    X = df[[f'lm_{i}_{j}' for i in range(5) for j in ['x', 'y', 'z']]].values

    # Select the label column
    y = df['label'].values

    # Reshape X to fit the model: (num_samples, 5 landmarks, 3 coordinates, 1)
    X = X.reshape(-1, 5, 3, 1)

    # Encode the labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Save label encoder for use in the virtual mouse and gesture recognition process
    with open('label_classes1.txt', 'w') as f:
        for class_name in label_encoder.classes_:
            f.write(f"{class_name}\n")

    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    # Preprocess the data and save it for further use
    csv_file = "hand_landmarks1.csv"  # Input file containing gesture data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(csv_file)

    # Save the preprocessed data as .npz file for training
    np.savez('preprocessed_data1.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Preprocessing complete. Data saved to preprocessed1_data.npz.")
