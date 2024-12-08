import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_colored_object(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, w, h]
    return None

def detect_object_in_hand(landmarks, object_bbox, frame_shape):
    if not landmarks or not object_bbox:
        return False

    frame_height, frame_width = frame_shape[:2]
    
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    hand_area = {
        'xmin': min(left_wrist.x, right_wrist.x) * frame_width - 50,
        'ymin': min(left_wrist.y, right_wrist.y) * frame_height - 50,
        'xmax': max(left_wrist.x, right_wrist.x) * frame_width + 50,
        'ymax': max(left_wrist.y, right_wrist.y) * frame_height + 50
    }

    x, y, w, h = object_bbox
    obj_center_x = x + w / 2
    obj_center_y = y + h / 2
    
    return (hand_area['xmin'] < obj_center_x < hand_area['xmax'] and
            hand_area['ymin'] < obj_center_y < hand_area['ymax'])

def main():
    video_path = 'v.mp4'
    cap = cv2.VideoCapture(video_path)
    
    lower_color = np.array([0, 100, 100])  # Example: Red color in HSV
    upper_color = np.array([10, 255, 255])

    object_in_hand = False
    release_counter = 0
    cooldown = 0
    cooldown_frames = 15  # Adjust this to set the cooldown period between releases

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        object_bbox = detect_colored_object(frame, lower_color, upper_color)

        if pose_results.pose_landmarks and object_bbox:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            x, y, w, h = object_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            current_object_in_hand = detect_object_in_hand(
                pose_results.pose_landmarks.landmark,
                object_bbox,
                frame.shape
            )

            # Detect object release
            if object_in_hand and not current_object_in_hand and cooldown == 0:
                release_counter += 1
                cooldown = cooldown_frames
                print(f"Object released! Count: {release_counter}")

            object_in_hand = current_object_in_hand

            if cooldown > 0:
                cooldown -= 1

        cv2.putText(frame, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Releases: {release_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()