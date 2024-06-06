from ultralytics import YOLO
import torch
import cv2

# Load the YOLO model
model = YOLO('best.pt')

# Specify the video source
video_source = "ikan/code/ikan-30-4detik.mp4"

# Open the video capture
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Use the YOLO model to predict on the frame
    results = model(frame)

    # Access the predictions and class names
    if isinstance(results, list) and results:
        predictions = results[0]
    else:
        predictions = []

    class_names = model.names if hasattr(model, 'names') else []

    # Get the number of fishes detected
    num_fishes = len(predictions)

    # Display the number of fishes at the top left corner of the frame
    cv2.putText(frame, f'Fishes: {num_fishes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw bounding boxes and confidence scores on the frame
    for pred in predictions:
        conf = pred[4].item()  # confidence score
        box = pred[:4].int().cpu().numpy()  # bounding box coordinates
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('YOLO Output', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
