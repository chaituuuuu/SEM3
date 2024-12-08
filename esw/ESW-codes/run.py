from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("path/to/best.pt")  # Replace with the actual path to your best.pt file

# Run inference on an image
results = model.predict("path/to/your/image.jpg", save=True)  # Replace with the path to your image

# Run inference on a video
# results = model.predict("path/to/your/video.mp4", save=True)  # Uncomment to use video

# Display the results
results.show()  # Display results in the notebook (only for images)
