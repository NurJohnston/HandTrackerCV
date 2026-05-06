import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

# ============================================================================
# FINGER LANDMARK INDICES
# ============================================================================

# Tip indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
FINGER_TIPS = [4, 8, 12, 16, 20]

# Base indices: Thumb(2), Index(5), Middle(9), Ring(13), Pinky(17)
FINGER_BASES = [2, 6, 10, 14, 18]


# ============================================================================
# FINGER COUNTING FUNCTION (Uses handedness for thumb detection)
# ============================================================================

def count_fingers(hand_landmarks, handedness):
    """
    Count how many fingers are raised.

    Parameters:
        hand_landmarks: MediaPipe hand landmark object
        handedness: "Left" or "Right" string from MediaPipe detection

    Returns:
        integer from 0-5 (number of raised fingers)
    """
    finger_count = 0

    # ================================================================
    # THUMB DETECTION (Index 4) - DIFFERENT FOR LEFT VS RIGHT HAND
    # ================================================================
    thumb_tip_x = hand_landmarks[4].x  # Point 4 (thumb tip)
    thumb_base_x = hand_landmarks[2].x  # Point 2 (thumb base)

    if handedness == "Right":
        # Right hand: thumb raised when tip is to the RIGHT of base
        if thumb_tip_x > thumb_base_x:
            finger_count += 1
    else:  # Left hand
        # Left hand: thumb raised when tip is to the LEFT of base
        if thumb_tip_x < thumb_base_x:
            finger_count += 1

    # ================================================================
    # INDEX, MIDDLE, RING, PINKY (SAME LOGIC FOR BOTH HANDS)
    # ================================================================
    # For these: tip should be ABOVE base (smaller y) to be raised

    # Index finger (tip: 8, base: 5)
    if hand_landmarks[8].y < hand_landmarks[6].y:
        finger_count += 1

    # Middle finger (tip: 12, base: 9)
    if hand_landmarks[12].y < hand_landmarks[10].y:
        finger_count += 1

    # Ring finger (tip: 16, base: 13)
    if hand_landmarks[16].y < hand_landmarks[14].y:
        finger_count += 1

    # Pinky finger (tip: 20, base: 17)
    if hand_landmarks[20].y < hand_landmarks[18].y:
        finger_count += 1

    return finger_count


# ============================================================================
# DRAWING FUNCTION
# ============================================================================

def draw_landmarks(frame, hand_landmarks):
    """Draw dots and lines for each landmark"""
    h, w, _ = frame.shape

    # Define connections between landmarks (bones)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    # Draw lines (bones) first
    for connection in connections:
        start_idx, end_idx = connection
        start_point = hand_landmarks[start_idx]
        end_point = hand_landmarks[end_idx]

        start_x = int(start_point.x * w)
        start_y = int(start_point.y * h)
        end_x = int(end_point.x * w)
        end_y = int(end_point.y * h)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    # Draw dots (landmarks) on top
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Download the model file if not present
    import urllib.request
    import os

    model_filename = "hand_landmarker.task"

    # Check if model file exists locally
    if not os.path.exists(model_filename):
        print("Downloading hand landmarker model...")
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(model_url, model_filename)
        print("Model downloaded successfully!")

    # Create hand landmarker
    try:
        base_options = python.BaseOptions(model_asset_path=model_filename)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,  # Allow detecting both hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        detector = vision.HandLandmarker.create_from_options(options)
        print("Hand detector initialized successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        detector.close()
        return

    print("Press 'q' to quit")
    print("Hold your hand up to the camera")
    print("Both left and right hands are supported!")

    while True:
        # Read frame from camera
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip horizontally (mirror effect)
        ##frame = cv2.flip(frame, 1)

        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        detection_result = detector.detect(mp_image)

        # Reset text position for multiple hands
        y_offset = 0

        # If hand detected
        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Get handedness (Left or Right) from detection result
                handedness = detection_result.handedness[i][0].category_name

                # Draw
                draw_landmarks(frame, hand_landmarks)

                # Count
                finger_count = count_fingers(hand_landmarks, handedness)

                # Get position for text (use wrist)
                h, w, _ = frame.shape
                wrist = hand_landmarks[0]

                # Offset each hand's text so they don't overlap
                text_x = int(wrist.x * w) - 50
                text_y = int(wrist.y * h) - 50 + y_offset
                y_offset += 80  # Move down for next hand

                # Ensure text stays inside frame
                if text_x < 10:
                    text_x = 10
                if text_y < 30:
                    text_y = 60

                # Display finger count with hand label
                cv2.putText(frame, f"{handedness} Hand: {finger_count} fingers",
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # No hand detected - show message
            cv2.putText(frame, "No hand detected", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the video feed
        cv2.imshow('Hand Finger Counter', frame)

        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Application closed.")


if __name__ == "__main__":
    main()