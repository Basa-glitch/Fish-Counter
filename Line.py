from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the model
model = YOLO("best.pt")

# Use webcam (usually, webcam is at index 0, but this might vary depending on your setup)
cap = cv2.VideoCapture(1)
assert cap.isOpened(), "Error accessing the webcam"

# Get frame dimensions and fps from the webcam
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
line_points = [(20, 400), (1080, 400)]

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)

    # Display the frame with counts
    cv2.imshow('Object Counting', im0)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
