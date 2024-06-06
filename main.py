from ultralytics import YOLO
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the model
model = YOLO("C:/Users/bryan/OneDrive/Desktop/Coding/Python/FishCounter/best.pt")

# Use external webcam (usually, external webcams are at index 1, but this might vary depending on your setup)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get frame dimensions and fps from the webcam
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
line_points = [(154, 192), (609, 192)]

# Function to count fish
def count_fish(tracks):
    return sum(1 for track in tracks if 'name' in track and track['name'] == 'fish')

# Function to draw bounding boxes
def draw_bounding_boxes(image, tracks):
    for track in tracks:
        if 'bbox' in track and 'name' in track:
            print(f"Drawing bbox for track: {track}")  # Debugging: Print track information
            x1, y1, x2, y2 = map(int, track['bbox'])  # Ensure coordinates are integers
            class_name = track['name']
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put the class name near the bounding box
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("Track missing 'bbox' or 'name'")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Failed to capture image from external webcam.")
        break

    # Track objects
    results = model.track(im0, persist=True, show=False)

    # Debugging: Print results structure
    print(f"Results: {results}")

    # Extract tracks based on the results structure
    if isinstance(results, dict) and 'track' in results:
        tracks = results['track']
    elif isinstance(results, list):
        tracks = results
    else:
        print("Unexpected results format.")
        tracks = []

    # Debugging: Print tracks to inspect their structure
    print(f"Tracks: {tracks}")

    # Get the count of detected fishes
    fish_count = count_fish(tracks)

    # Draw bounding boxes around detected objects
    draw_bounding_boxes(im0, tracks)

    # Overlay fish count on the image
    cv2.putText(im0, f'Fish Count: {fish_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with counts
    cv2.imshow('Object Counting', im0)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
