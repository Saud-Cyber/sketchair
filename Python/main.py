import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to access the camera.")
        break

    # Display the camera feed
    cv2.imshow("Air Drawing - Level 1", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()