import cv2
import mediapipe as mp
import numpy as np
import os

# --- Configuration ---
# Define the gestures you want to collect data for
GESTURES = ["open_palm", "fist", "thumbs_up", "pointing_up"]
DATA_PATH = "data" # Folder to save the collected data

# --- Setup ---
# Create the main data directory if it doesn't exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Create subdirectories for each gesture
for gesture in GESTURES:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

print(f"Data will be saved in the '{DATA_PATH}' directory.")
print("Press the first letter of a gesture to start collecting samples.")
print("Press 'q' to quit.")
print("-" * 30)
for gesture in GESTURES:
    print(f"Press '{gesture[0]}' to collect data for '{gesture}'")
print("-" * 30)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Collect data for one hand at a time for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(rgb_image)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Project-0 - Data Collection', image)

    # --- Data Collection Logic ---
    key = cv2.waitKey(10) & 0xFF

    # Quit if 'q' is pressed
    if key == ord('q'):
        break

    # Check if the pressed key corresponds to a gesture
    for gesture in GESTURES:
        if key == ord(gesture[0]):
            print(f"Collecting sample for: {gesture}")
            
            # Ensure a hand is detected
            if results.multi_hand_landmarks:
                # Extract landmarks from the first detected hand
                landmarks = results.multi_hand_landmarks[0]
                
                # Flatten the landmark data (21 landmarks * 3 coordinates = 63 values)
                landmark_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
                
                # Find the next available file number
                gesture_path = os.path.join(DATA_PATH, gesture)
                sample_num = len(os.listdir(gesture_path))
                file_path = os.path.join(gesture_path, f'{sample_num}.npy')
                
                # Save the landmark data as a .npy file
                np.save(file_path, landmark_data)
                print(f"Saved sample {sample_num} to {file_path}")
            else:
                print("No hand detected! Please show your hand to the camera.")
            
            # Add a small delay to prevent multiple captures from one key press
            cv2.waitKey(500) 


# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Data collection finished. Resources released.")
