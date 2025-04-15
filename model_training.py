import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
import visualkeras

def create_model(num_classes):
    # Build a CNN model
    model = Sequential([
        # First convolutional layer
        Conv2D(64, (2, 2), activation='relu', input_shape=(5, 3, 1), padding='same'),
        MaxPooling2D(pool_size=(2, 2)),  # Reducing the dimensions by half in both directions

        # Second convolutional layer
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 1)),  # Avoid reducing the dimensions to zero in one direction

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),

        # Output layer with 'num_classes' classes
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Load preprocessed data
    data = np.load('preprocessed_data.npz')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    # Create and train the CNN model
    model = create_model(num_classes=len(np.unique(y_train)))

    # Display the model summary
    model.summary()
    # Train the model and save the training history
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)
    visualkeras.layered_view(model).show()
    # Save the trained model
    model.save('gesture_recognition_model8.keras')
    print("Model trained and saved as 'gesture_recognition_model8.keras'.")

    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 4))
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()



