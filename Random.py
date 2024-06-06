import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # Maximum number of hands to detect
    min_detection_confidence=0.5,  # Detection confidence threshold
    min_tracking_confidence=0.5)  # Tracking confidence threshold
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks on the image

# Access the webcam.
cap = cv2.VideoCapture(0)

# Get the width and height of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the border
border_x = frame_width // 2

while True:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display.
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and find hands.
    results = hands.process(image)

    # Convert the image color back so it can be displayed.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the border
    cv2.line(image, (border_x, 0), (border_x, frame_height), (255, 0, 0), 2)

    # Draw the hand annotations on the image.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the x-coordinate of the first landmark (wrist)
            hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame_width)

            # Check which side of the border the hand is on
            if hand_x < border_x:
                side_text = "Left"
            else:
                side_text = "Right"

            # Display which side the hand is on
            cv2.putText(image, side_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', image)

    # Press 'q' to close the window
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
