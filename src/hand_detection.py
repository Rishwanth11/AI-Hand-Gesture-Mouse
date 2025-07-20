import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please ensure your webcam is connected and not in use by another application.")
        return

    webcam_width = 640
    webcam_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

    screen_width, screen_height = pyautogui.size()
    print(f"Screen Resolution: {screen_width}x{screen_height}")

    horizontal_frame_reduction = 10
    vertical_frame_reduction = 10

    tracking_region_x1 = horizontal_frame_reduction
    tracking_region_y1 = vertical_frame_reduction
    tracking_region_x2 = webcam_width - horizontal_frame_reduction
    tracking_region_y2 = webcam_height - vertical_frame_reduction

    smoothening = 8

    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    click_threshold = 30

    click_debounce_time = 0.3
    last_click_time = 0

    is_pinching = False

    calibration_mode = True
    calib_point1 = None
    calib_point2 = None
    calib_stage = 1

    print("\n--- Hand Mouse Controller ---")
    print("Welcome! Let's calibrate your tracking area first.")
    print("Instructions:")
    print("  - Place your hand (index finger tip) at the desired TOP-LEFT corner of your control area.")
    print("  - Pinch your thumb and index finger together to confirm this point.")
    print("  - Then, do the same for the BOTTOM-RIGHT corner.")
    print("  - Press 's' to skip calibration and use the default tracking region.")
    print("  - Press 'q' to quit at any time.")


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting...")
            break

        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * webcam_width), int(lm.y * webcam_height)
                lmList.append([id, cx, cy])

                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

            index_x, index_y = lmList[8][1], lmList[8][2]
            thumb_x, thumb_y = lmList[4][1], lmList[4][2]

            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
            cv2.putText(img, f"Dist: {int(distance)}", (thumb_x + 50, thumb_y),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        else:
            cv2.putText(img, "No Hand Detected", (webcam_width // 2 - 100, webcam_height // 2),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


        if calibration_mode:
            current_time = time.time()

            if calib_stage == 1:
                cv2.putText(img, "Calib Stage 1: TOP-LEFT", (webcam_width//2 - 180, webcam_height//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                if hand_detected and distance < click_threshold:
                    if not is_pinching and (current_time - last_click_time > click_debounce_time):
                        calib_point1 = (index_x, index_y)
                        calib_stage = 2
                        last_click_time = current_time
                        is_pinching = True
                        print(f"Calibrated Point 1: {calib_point1}")
                elif hand_detected:
                    is_pinching = False
                if calib_point1:
                    cv2.circle(img, calib_point1, 15, (0, 255, 0), cv2.FILLED)

            elif calib_stage == 2:
                cv2.putText(img, "Calib Stage 2: BOTTOM-RIGHT", (webcam_width//2 - 220, webcam_height//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                if hand_detected and distance < click_threshold:
                    if not is_pinching and (current_time - last_click_time > click_debounce_time):
                        calib_point2 = (index_x, index_y)
                        tracking_region_x1 = min(calib_point1[0], calib_point2[0])
                        tracking_region_y1 = min(calib_point1[1], calib_point2[1])
                        tracking_region_x2 = max(calib_point1[0], calib_point2[0])
                        tracking_region_y2 = max(calib_point1[1], calib_point2[1])

                        calib_stage = 3
                        last_click_time = current_time
                        is_pinching = True
                        print(f"Calibrated Point 2: {calib_point2}")
                elif hand_detected:
                    is_pinching = False
                if calib_point2:
                    cv2.circle(img, calib_point2, 15, (0, 255, 0), cv2.FILLED)

            elif calib_stage == 3:
                cv2.putText(img, "Calibration Complete!", (webcam_width//2 - 180, webcam_height//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
                if time.time() - last_click_time > 1:
                    calibration_mode = False
                    print(f"Calibration successful! Tracking Region: {tracking_region_x1, tracking_region_y1} to {tracking_region_x2, tracking_region_y2}")

        else:
            cv2.rectangle(img, (tracking_region_x1, tracking_region_y1),
                          (tracking_region_x2, tracking_region_y2),
                          (255, 0, 255), 2)

            if hand_detected:
                current_time = time.time()

                if distance < click_threshold:
                    if not is_pinching:
                        is_pinching = True
                        if current_time - last_click_time > click_debounce_time:
                            try:
                                pyautogui.click()
                                print(f"Click detected! Distance: {int(distance)}")
                                cv2.circle(img, (index_x, index_y), 30, (0, 255, 255), cv2.FILLED)
                                last_click_time = current_time
                            except pyautogui.FailSafeException:
                                print("Click interrupted by Fail-Safe. Exiting...")
                                break
                    else:
                        is_pinching = False

                if (not is_pinching):
                    if tracking_region_x1 < index_x < tracking_region_x2 and \
                       tracking_region_y1 < index_y < tracking_region_y2:

                        mapped_mouse_x = int(np.interp(index_x,
                                                       (tracking_region_x1, tracking_region_x2),
                                                       (0, screen_width)))
                        mapped_mouse_y = int(np.interp(index_y,
                                                       (tracking_region_y1, tracking_region_y2),
                                                       (0, screen_height)))

                        curr_x = prev_x + (mapped_mouse_x - prev_x) / smoothening
                        curr_y = prev_y + (mapped_mouse_y - prev_y) / smoothening

                        try:
                            pyautogui.moveTo(curr_x, curr_y)
                        except pyautogui.FailSafeException:
                            print("Mouse movement interrupted by Fail-Safe. Move physical mouse to corner to disable.")
                            break

                        prev_x, prev_y = curr_x, curr_y

                        cv2.circle(img, (int(curr_x), int(curr_y)), 10, (0, 255, 0), cv2.FILLED)

                    else:
                        cv2.putText(img, "Hand Out of Bounds!", (tracking_region_x1, tracking_region_y1 - 20),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


        cv2.imshow("Hand Mouse Control", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = True
            calib_stage = 1
            calib_point1 = None
            calib_point2 = None
            print("\n--- Calibration Restarted ---")
            print("Place hand at TOP-LEFT corner for tracking region.")
        elif key == ord('s') and calibration_mode:
            calibration_mode = False
            tracking_region_x1 = horizontal_frame_reduction
            tracking_region_y1 = vertical_frame_reduction
            tracking_region_x2 = webcam_width - horizontal_frame_reduction
            tracking_region_y2 = webcam_height - vertical_frame_reduction
            print("Skipping calibration. Using default tracking region.")


    cap.release()
    cv2.destroyAllWindows()
    print("Hand Mouse Control application stopped.")

if __name__ == "__main__":
    main()

    #WITH THE HELP OF CHAT GPT
    # so this will perform basic operations like moving cursor and selecting anything