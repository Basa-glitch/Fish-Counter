import cv2

def get_fps(camera):
    # Start capturing frames
    cap = cv2.VideoCapture(camera)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get the frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the camera
    cap.release()

    return fps

# Get the fps of the default camera (usually webcam)
fps = get_fps(0)  # Pass 0 for default camera
print("FPS of default camera:", fps)

# If you have multiple cameras, you can specify the camera index
# Replace 1 with the index of your external webcam if it's not the default camera
external_webcam_fps = get_fps(1)
print("FPS of external webcam:", external_webcam_fps)
