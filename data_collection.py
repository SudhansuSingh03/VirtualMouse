import cv2
import mediapipe as mp
import pandas as pd

# Define the list of gestures and their instructions
GESTURES = {
    'move': "Use the index finger for cursor movement",
    'left-click': "Use index and middle fingers for left-click",
    'right-click': "Use index, middle, and ring fingers for right-click",
    'hold': "Close all fingers for hold",
    'stop': "Open all fingers to stop the cursor"
}


def collect_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    gesture_data = []
    gesture_labels = []

    print("Starting data collection for virtual mouse gestures.")
    print("Follow the on-screen instructions for each gesture.")

    # Go through each gesture in the GESTURES dictionary
    for gesture, instruction in GESTURES.items():
        print(f"\nNext Gesture: {gesture}")
        print(f"Instruction: {instruction}")
        print("Press 's' to start recording this gesture, or 'q' to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = hands.process(rgb_frame)

            # Draw hand landmarks if detected
            if output.multi_hand_landmarks:
                for hand_landmarks in output.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Show the current frame
            cv2.imshow('Hand Tracking', frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting data collection.")
                cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord('s'):
                print(f"Recording {gesture}. Press 'q' to stop recording.")

                # Begin recording for this gesture
                while True:
                    success, frame = cap.read()
                    if not success:
                        print("Failed to capture frame. Exiting...")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    output = hands.process(rgb_frame)

                    # Draw landmarks if hands are detected
                    if output.multi_hand_landmarks:
                        for hand_landmarks in output.multi_hand_landmarks:
                            landmarks = hand_landmarks.landmark

                            # Collect key landmarks for all fingers
                            key_landmarks = [
                                landmarks[8],  # Index finger tip
                                landmarks[12],  # Middle finger tip
                                landmarks[16],  # Ring finger tip
                                landmarks[4],  # Thumb tip
                                landmarks[0]  # Wrist
                            ]

                            # Flatten key landmarks into a list of 15 values (5 points * 3 coordinates)
                            landmark_positions = []
                            for lm in key_landmarks:
                                landmark_positions.extend([lm.x, lm.y, lm.z])

                            if len(landmark_positions) == 15:
                                gesture_data.append(landmark_positions)
                                gesture_labels.append(gesture)

                        # Draw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                    # Show the frame
                    cv2.imshow('Hand Tracking', frame)

                    # Press 'q' to stop recording the current gesture and proceed to the next one
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"Stopped recording {gesture}. Moving to the next gesture.")
                        break
                break  # Exit the inner loop to move to the next gesture

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data into a CSV file
    if len(gesture_data) > 0:
        columns = [f'lm_{i}_{j}' for i in range(5) for j in ['x', 'y', 'z']]
        df = pd.DataFrame(gesture_data, columns=columns)
        df['label'] = gesture_labels

        try:
            df.to_csv('hand_landmarks1.csv', index=False)
            print(f"Data saved to hand_landmarks1.csv with {len(gesture_data)} samples.")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
    else:
        print("No hand landmarks detected or data collected.")


if __name__ == "__main__":
    collect_data()
