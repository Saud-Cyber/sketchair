import cv2
import mediapipe as mp

# --------------------------------
# MediaPipe Hand Landmarker Setup
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# --------------------------------
# Open Webcam
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to access the camera.")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    result = landmarker.detect(mp_image)

    # --------------------------------
    # Index Finger Tracking
    # --------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Landmark 8 = Index fingertip
        index_finger = hand[8]

        # Convert normalized coordinates to pixels
        x = int(index_finger.x * frame.shape[1])
        y = int(index_finger.y * frame.shape[0])

        # Draw fingertip
        cv2.circle(
            frame,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        # Display coordinates
        cv2.putText(
            frame,
            f"X: {x}  Y: {y}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # --------------------------------
    # Display
    # --------------------------------

    cv2.imshow("Air Drawing - Level 3", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------
# Cleanup
# --------------------------------

camera.release()
landmarker.close()
cv2.destroyAllWindows()