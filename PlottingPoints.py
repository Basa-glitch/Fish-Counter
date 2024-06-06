import cv2

# Try these coordinates for the line
# X: 154, Y: 192
# X: 609, Y: 202

# Initialize a list to store click points and a variable for the current mouse position
click_points = []
mouse_position = (0, 0)

def click_event(event, x, y, flags, params):
    global click_points, mouse_position

    # Update mouse position
    mouse_position = (x, y)

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Left Click - X: {x}, Y: {y}')
        click_points.append((x, y))

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        print(f'Right Click - X: {x}, Y: {y}')
        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        cv2.putText(frame, f'B:{b}, G:{g}, R:{r}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def draw_with_mouse():
    global frame
    if len(click_points) > 0:
        # Draw all segments
        for i in range(1, len(click_points)):
            cv2.line(frame, click_points[i - 1], click_points[i], (0, 255, 0), 2)
        # Draw line from the last click point to the current mouse position
        cv2.line(frame, click_points[-1], mouse_position, (0, 255, 0), 2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)  # Adjust the device number as needed

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', click_event)

    while True:
        ret, frame_original = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = frame_original.copy()
        draw_with_mouse()
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
