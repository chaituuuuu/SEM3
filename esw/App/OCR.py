import cv2
import os
import easyocr
import re
from datetime import datetime

def crop_image(frame, bounding_box):
    x, y, w, h = bounding_box
    height, width = frame.shape[:2]
    
    # Ensure the bounding box is within the frame's dimensions
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("Bounding box is out of frame bounds")
    
    return frame[y:y+h, x:x+w]

def extract_timestamp_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        return

    timestamp_bounding_boxes = [(70, 1160, 545, 120)]  #  video-1:  80, 1120, 505, 160
    reader = easyocr.Reader(['en'])
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    frames_per_minute = int(frame_rate * 60)  # Calculate frames per minute
    frame_count = 0
    
    # Create output directories
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    timestamp_directory = f"{video_name}_timestamp_frames"  # For frames with timestamps
    no_timestamp_directory = f"{video_name}_no_timestamp_frames"  # For frames without timestamps
    
    # Create both directories
    os.makedirs(timestamp_directory, exist_ok=True)
    os.makedirs(no_timestamp_directory, exist_ok=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            frame_count += 1
            
            # Extract frame at desired frame rate (1 frame per minute)
            if frame_count % frames_per_minute == 0:
                try:
                    for timestamp_bounding_box in timestamp_bounding_boxes:
                        # Crop the region where timestamp is located
                        try:
                            timestamp_region = crop_image(frame, timestamp_bounding_box)
                        except ValueError as e:
                            print(f"Skipping bounding box {timestamp_bounding_box}: {e}")
                            continue

                        # Draw a red bounding box on the timestamp region
                        x, y, w, h = timestamp_bounding_box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # cv2.imshow('timestamp', timestamp_region)

                        # Read text from the timestamp region
                        results = reader.readtext(timestamp_region, allowlist= '0123456789-: ')
                        print(f"results {results}")
                        
                        timestamp = None
                        full_text = ''
                        # Look for text matching datetime format
                        for (bbox, text, prob) in results:
                            # Clean the text and look for datetime pattern
                            text = text.strip()
                            full_text = full_text + ' ' + text
                            datetime_pattern = r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}'
                            match = re.search(datetime_pattern, full_text)
                            
                            if match:
                                timestamp = match.group()
                                print(f"timestamp {timestamp}")
                                break
                        else:
                            continue
                        break
                        
                    #Show the frame
                    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow("frame", 800, 600)
                    # cv2.imshow("frame", frame)
                    
                    if timestamp:
                        # Save frame with timestamp in timestamp directory
                        clean_timestamp = timestamp.replace(':', '-').replace(' ', '_')
                        output_file = f"{timestamp_directory}/{clean_timestamp}.jpg"
                        cv2.imwrite(output_file, frame)
                        print(f"Timestamp detected! Frame saved as {output_file}")
                    else:
                        # Save frame without timestamp in no_timestamp directory
                        output_file = f"{no_timestamp_directory}/frame_{frame_count}.jpg"
                        cv2.imwrite(output_file, frame)
                        print(f"No timestamp detected. Frame saved as {output_file}")
                    
                    # Wait for a key press and exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        # Print summary
        timestamp_count = len(os.listdir(timestamp_directory))
        no_timestamp_count = len(os.listdir(no_timestamp_directory))
        print("\nProcessing Complete!")
        print(f"Frames with timestamps: {timestamp_count}")
        print(f"Frames without timestamps: {no_timestamp_count}")
        print(f"Total frames processed: {timestamp_count + no_timestamp_count}")
        
        cap.release()
        cv2.destroyAllWindows()

def process_videos_in_folder(folder_path):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # Add more extensions if needed
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):
            video_file = os.path.join(folder_path, filename)
            print(f"Processing video: {video_file}")
            extract_timestamp_frames(video_file)

if __name__ == "__main__":
    folder_path = "GVP-Video"  # Replace with your folder path
    process_videos_in_folder(folder_path)