import cv2
import numpy as np
from datetime import datetime

def count_objects(frame, min_area=100):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Apply Canny edge detection for better detection of objects
    edges = cv2.Canny(blurred_frame, 50, 150)
    
    # Use morphological operations to close small gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours (i.e., object boundaries)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours to reduce noise (min area of 'min_area' pixels)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return len(large_contours), closed_edges, large_contours

def process_video(video_path, output_folder, diff_threshold=30, min_area=100):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a black background of the same size as the video frame
    black_background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return
    
    prev_object_count, _, _ = count_objects(prev_frame, min_area=min_area)
    
    for frame_index in range(1, frame_count, 5):
        ret, current_frame = cap.read()
        if not ret:
            break
        
        # Count objects in the current frame
        current_object_count, binary_frame, large_contours = count_objects(current_frame, min_area=min_area)
        
        # Calculate the difference in object count between frames
        object_count_diff = abs(current_object_count - prev_object_count)
        
        if object_count_diff > 0:  # Check if there's a significant change
            # Create a mask for the current frame
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.drawContours(mask, large_contours, -1, (255), thickness=cv2.FILLED)  # Fill contours in mask
            
            # Extract the colored objects from the original frame
            colored_objects = cv2.bitwise_and(current_frame, current_frame, mask=mask)
            
            # Create a black result frame and place the colored objects on it
            black_result_frame = black_background.copy()
            black_result_frame[mask > 0] = colored_objects[mask > 0]
            
            # Add timestamp, frame index, and object count
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(black_result_frame, f"Frame: {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(black_result_frame, f"Objects: {current_object_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(black_result_frame, f"Time: {timestamp}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save the result with objects on black background
            output_path = f"{output_folder}/frame_diff_{frame_index:04d}.png"
            cv2.imwrite(output_path, black_result_frame)
            
            print(f"Significant change detected in frame {frame_index}: {object_count_diff} object(s) difference")
        else:
            print(f"No significant change in frame {frame_index}")
        
        # Update previous frame object count
        prev_object_count = current_object_count
    
    # Release the video capture object
    cap.release()
    print("Video processing completed")

# Usage example
video_path = "v.mp4"
output_folder = "new"
diff_threshold = 30  # Adjust this value to change sensitivity
min_area = 100       # Adjust this value to detect smaller objects
process_video(video_path, output_folder, diff_threshold, min_area)
