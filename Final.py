import cv2
from ultralytics import YOLO
from supervision.tools.line_counter import LineCounter
import serial
from datetime import datetime

# Setup serial communication (Check the right COM port and baudrate)
ser = serial.Serial('COM5', 9600, timeout=1)

# Load your model
model_path = 'C:/Users/bryan/OneDrive/Desktop/Coding/Python/FishCounter/best.pt'
model = YOLO(model_path)

# Line coordinates
line_start = (71, 400)
line_end = (526, 400)
line_y = line_start[1]

line_counter = LineCounter(start=line_start, end=line_end)

# Initialize variables
Counter = 0
previous_positions = {}
counted_objects = {}

# Function to check if a fish has crossed the line
def has_crossed_line(center_y, previous_y):
    return (previous_y < line_y and center_y >= line_y) or (previous_y > line_y and center_y <= line_y)

# Video capture from external webcam (change 1 to the appropriate index for your external webcam)
cap = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = "C:/Users/bryan/OneDrive/Desktop/Coding/Python/FishCounter/Recordings/fish_counter_output.avi"
out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


def send_Counter_to_arduino():
    global Counter
    data_to_send = f"{Counter}\n".encode()
    ser.write(data_to_send)
    print(f"Sent to Arduino: {Counter}")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame, verbose=False)
    
    # Draw the line on the frame
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # Process detections
    for result_index, result in enumerate(results):
        for detection_index, detection in enumerate(result.boxes):
            bbox = detection.xyxy[0].tolist()  # Bounding box coordinates
            conf = detection.conf[0].item()  # Confidence score
            class_id = int(detection.cls[0].item())  # Class ID

            # Filter out detections of non-fish objects (if your model is multi-class)
            if class_id != 0:  # Assuming class_id 0 is for fish
                continue

            # Calculate center of the bounding box
            x1, y1, x2, y2 = map(int, bbox)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw the bounding box and center point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Generate unique identifier for the detected object
            object_id = f"{class_id}-{result_index}-{detection_index}"

            # Track fish positions and check for line crossing
            if object_id not in previous_positions:
                previous_positions[object_id] = center_y
                counted_objects[object_id] = False  # Mark as not counted
            else:
                previous_y = previous_positions[object_id]
                if has_crossed_line(center_y, previous_y) and not counted_objects[object_id]:
                    Counter += 1
                    send_Counter_to_arduino()
                    counted_objects[object_id] = True  # Mark as counted
                    print(f"Fish Counted: {Counter}")


            # Update the previous position
            previous_positions[object_id] = center_y

    # Display the counter on the frame
    cv2.putText(frame, f'Fish: {len(results[0].boxes)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Counter: {Counter}', (frame.shape[1] - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2, cv2.LINE_AA)

    # Display date and time at the bottom right corner
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_size = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = frame.shape[0] - 10
    cv2.putText(frame, current_time, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('Fish Counter', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
