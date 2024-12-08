from inference_sdk import InferenceHTTPClient
import cv2
import tempfile
import os

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="u3NpMO9V1kTHv2REIkTi"
)

# Initialize the video capture
video_path = "hello.mp4"
cap = cv2.VideoCapture(video_path)  # Load video file

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Directories for saving images and coordinates
image_dir = "output_images"
coord_dir = "output_coordinates"

# Create directories if they don't exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(coord_dir, exist_ok=True)

# Confidence threshold
confidence_threshold = 0.4

# Calculate frame skip rate for 3 FPS (process 1 frame for every (fps // 3) frames)
frame_skip_rate = fps // 3

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Skip frames to process 3 FPS
    if frame_idx % frame_skip_rate != 0:
        frame_idx += 1
        continue

    # Create a temporary file to save the frame as an image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
        temp_image_path = temp_image.name
        cv2.imwrite(temp_image_path, frame)

        # Perform inference on the temporary image file
        result = CLIENT.infer(temp_image_path, model_id="final-final-3/1")

    # Define image and coordinate file paths
    image_file_path = os.path.join(image_dir, f"frame_{frame_idx}.jpg")
    coord_file_path = os.path.join(coord_dir, f"frame_{frame_idx}.txt")

    # Save the frame image
    cv2.imwrite(image_file_path, frame)

    # Prepare the coordinates to save
    bounding_boxes = []
    for prediction in result.get("predictions", []):
        confidence = prediction["confidence"]
        if confidence < confidence_threshold:
            continue

        # Get the coordinates of the bounding box
        x, y, width_box, height_box = prediction["x"], prediction["y"], prediction["width"], prediction["height"]

        # Normalize the bounding box coordinates to the range [0, 1]
        norm_x = x / width
        norm_y = y / height
        norm_width = width_box / width
        norm_height = height_box / height

        # Prepare the bounding box information to be written
        # First `0` represents the waste type (hardcoded as 0)
        bounding_boxes.append(f"0 {norm_x:.4f} {norm_y:.4f} {norm_width:.4f} {norm_height:.4f}")

    # Save the bounding box coordinates to a text file
    with open(coord_file_path, "w") as coord_file:
        for bbox in bounding_boxes:
            coord_file.write(f"{bbox}\n")

    # Clean up the temporary file after processing the frame
    os.remove(temp_image_path)

    frame_idx += 1

# Release the video capture object
cap.release()

print(f"Images saved to: {image_dir}")
print(f"Coordinates saved to: {coord_dir}")
