import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import tkinter as tk
from threading import Thread

# Initialize MediaPipe Hand and pyautogui for controlling the mouse
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the trained gesture recognition model
model = load_model('gesture_recognition_model7.keras')

# Load label classes from file
label_classes = []
with open('label_classes1.txt', 'r') as f:
    label_classes = f.read().splitlines()

# Gesture-to-action map
gesture_map = {
    'move': 'move',
    'left-click': 'left_click',
    'right-click': 'right_click',
    'stop': 'stop',
    'hold': 'hold'
}

# Initialize camera with higher frame rate
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Get screen size for mapping mouse movement
screen_width, screen_height = pyautogui.size()

# Global flag to control the camera
camera_active = False


def predict_gesture(landmarks, model):
    """Predict gesture based on landmarks using the trained model."""
    landmarks = np.array(landmarks).reshape(-1, 5, 3, 1)  # Reshape for CNN input (5 landmarks, x, y, z)
    prediction = model.predict(landmarks)
    predicted_index = np.argmax(prediction)
    return label_classes[predicted_index]


def perform_mouse_action(gesture, x, y):
    """Map gestures to pyautogui mouse actions."""
    if gesture == 'move':
        pyautogui.moveTo(x, y, duration=0.05)  # Adding a small delay for smoother movement
    elif gesture == 'left-click':
        pyautogui.click(button='left')
    elif gesture == 'right-click':
        pyautogui.click(button='right')
    elif gesture == 'hold':
        pyautogui.mouseDown()
    elif gesture == 'stop':
        pyautogui.mouseUp()


# Smoothing mouse movement
prev_x, prev_y = 0, 0


def smooth_movement(x, y):
    global prev_x, prev_y
    smoothed_x = (prev_x + x) / 2
    smoothed_y = (prev_y + y) / 2
    prev_x, prev_y = smoothed_x, smoothed_y
    return int(smoothed_x), int(smoothed_y)


def start_camera():
    global camera_active
    camera_active = True
    Thread(target=run_camera).start()


def stop_camera():
    global camera_active
    camera_active = False


def run_camera():
    global camera_active
    last_time = time.time()

    while camera_active:
        success, frame = cap.read()
        if not success or not camera_active:
            break

        # Flip the frame horizontally for a natural feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect key landmarks (index, middle, ring, thumb, and wrist)
                landmarks = [
                    hand_landmarks.landmark[8],  # Index finger tip
                    hand_landmarks.landmark[12],  # Middle finger tip
                    hand_landmarks.landmark[16],  # Ring finger tip
                    hand_landmarks.landmark[4],  # Thumb tip
                    hand_landmarks.landmark[0]  # Wrist
                ]

                # Prepare landmarks for gesture prediction
                landmark_positions = [[lm.x, lm.y, lm.z] for lm in landmarks]

                # Predict gesture based on landmarks
                gesture = predict_gesture(landmark_positions, model)
                print(f"Predicted Gesture: {gesture}")

                # Draw landmarks on the frame for visualization
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Perform mouse action based on the predicted gesture
                if landmark_positions:
                    # Use the index finger's tip as the mouse pointer
                    finger_x, finger_y = landmarks[0].x * w, landmarks[0].y * h
                    mouse_x = int(finger_x * screen_width / w)
                    mouse_y = int(finger_y * screen_height / h)

                    # Smooth the movement
                    smooth_x, smooth_y = smooth_movement(mouse_x, mouse_y)

                    # Perform mouse action based on the predicted gesture
                    perform_mouse_action(gesture, smooth_x, smooth_y)

        # Calculate and display the frame rate
        curr_time = time.time()
        fps = 1 / (curr_time - last_time)
        last_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Show the frame with drawn landmarks
        cv2.imshow("Virtual Mouse", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def close_program():
    """Safely close the camera and exit the program."""
    stop_camera()  # Stop the camera if it's running
    cap.release()  # Release the camera resource
    root.destroy()  # Close the Tkinter window
    print("Program closed.")


# Tkinter GUI for Start/Stop/Close buttons
root = tk.Tk()
root.title("Virtual Mouse Control")

start_button = tk.Button(root, text="Start Camera", command=start_camera, width=20, height=2)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Camera", command=stop_camera, width=20, height=2)
stop_button.pack(pady=10)

close_button = tk.Button(root, text="Close Program", command=close_program, width=20, height=2)
close_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()

# Release the webcam when the program is done
cap.release()
