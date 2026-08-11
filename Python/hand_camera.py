import cv2
import mediapipe as mp
import time


# ==========================================
# MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Actual camera resolution: {actual_width} x {actual_height}")

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()


# ==========================================
# TIMESTAMP
# ==========================================

start_time = time.monotonic()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        continue


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp = int(
        (time.monotonic() - start_time) * 1000
    )


    # Detect
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ==========================================
    # DRAW ALL LANDMARKS
    # ==========================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]


        for start, end in connections:

            x1 = int(
                hand[start].x *
                frame.shape[1]
            )

            y1 = int(
                hand[start].y *
                frame.shape[0]
            )

            x2 = int(
                hand[end].x *
                frame.shape[1]
            )

            y2 = int(
                hand[end].y *
                frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


        # ======================================
        # LANDMARK POINTS
        # ======================================

        for i, landmark in enumerate(hand):

            x = int(
                landmark.x *
                frame.shape[1]
            )

            y = int(
                landmark.y *
                frame.shape[0]
            )


            # Index fingertip = RED
            if i == 8:

                cv2.circle(
                    frame,
                    (x, y),
                    12,
                    (0, 0, 255),
                    -1
                )

            else:

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # Landmark number
            cv2.putText(
                frame,
                str(i),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


        # ======================================
        # INDEX TIP INFO
        # ======================================

        index = hand[8]

        ix = int(index.x * frame.shape[1])
        iy = int(index.y * frame.shape[0])


        cv2.putText(
            frame,
            f"INDEX TIP (8): {ix}, {iy}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "HAND LANDMARK DIAGNOSTIC",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

landmarker.close()

cv2.destroyAllWindows()