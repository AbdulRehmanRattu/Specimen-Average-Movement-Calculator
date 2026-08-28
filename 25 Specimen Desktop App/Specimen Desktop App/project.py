import cv2
import numpy as np

def calculate_movement(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    total_movement = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        diff = cv2.absdiff(prev_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        movement = np.sum(threshold_diff) / 255
        total_movement += movement
        frame_count += 1

        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

        prev_frame = frame

    average_movement = total_movement / frame_count
    print(f"Average Movement: {average_movement}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "sperm.mp4"
    calculate_movement(video_path)
