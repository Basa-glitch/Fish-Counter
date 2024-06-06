import serial
import time
import cv2
import mediapipe as mp
import threading

# Setup serial communication (Check the right COM port and baudrate)
ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # Wait for the serial connection to initialize

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Use the appropriate webcam index

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Counter variable
counter = 0

# Flag to indicate if hand has crossed the line
hand_crossed_line = False

# Lock for synchronizing access to the counter variable
counter_lock = threading.Lock()

def send_counter_to_arduino():
    global counter
    while True:
        with counter_lock:
            ser.write(f"{counter}\n".encode())
        time.sleep(1)  # Send the counter value to Arduino every 1 second

# Start the thread for sending data to Arduino
threading.Thread(target=send_counter_to_arduino, daemon=True).start()

while True:
    # Webcam capture part
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the image
    results = hands.process(rgb_frame)

    # Draw bounding boxes around the detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the landmarks for the hand
            landmarks = hand_landmarks.landmark

            # Get the coordinates of the tip of the index finger (landmark 8)
            tip_index_x = int(landmarks[8].x * frame.shape[1])
            tip_index_y = int(landmarks[8].y * frame.shape[0])

            # Check if the tip of the index finger crosses the line
            if tip_index_x >= 326:
                # If hand has not crossed the line before, increment the counter
                if not hand_crossed_line:
                    with counter_lock:
                        counter += 1
                    print("Counter:", counter)
                    hand_crossed_line = True
            else:
                # Reset the flag if hand moves back
                hand_crossed_line = False

    # Draw a line on the frame
    cv2.line(frame, (326, 12), (326, 468), (255, 0, 0), 2)  # Blue line with 2 pixels thickness

    # Display the counter value on the frame
    with counter_lock:
        cv2.putText(frame, "Counter: " + str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)  # Dark orange color

    # Display the webcam frame
    cv2.imshow('Webcam Feed', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
