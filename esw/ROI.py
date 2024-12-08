import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_bounding_boxes(file_path):
    """
    Load bounding boxes from a file (assuming Roboflow format: class, x_center, y_center, width, height).
    """
    bounding_boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:
                _, x_center, y_center, width, height = map(float, data[:5])
                bounding_boxes.append((x_center, y_center, width, height))
    return bounding_boxes


def process_image(image_path, bounding_box_file, overlap_threshold=0.5):
    """
    Process the image and determine the cleanliness of each segment.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Load bounding boxes
    bounding_boxes = load_bounding_boxes(bounding_box_file)

    # Convert bounding boxes to absolute coordinates
    absolute_boxes = []
    for x_center, y_center, width, height in bounding_boxes:
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)
        absolute_boxes.append((x1, y1, x2, y2))

    # Divide the image into 8 segments (2x4 grid)
    rows, cols = 12, 8
    segment_width = image_width // cols
    segment_height = image_height // rows

    clean_segments = []

    # for i in range(8,rows-2):
    #     for j in range(1,cols-1):
    for i in range(rows):
        for j in range(cols):
            x1 = j * segment_width
            y1 = i * segment_height
            x2 = x1 + segment_width
            y2 = y1 + segment_height
            segment_area = segment_width * segment_height

            # Calculate intersection area with bounding boxes
            intersection_area = 0
            for bx1, by1, bx2, by2 in absolute_boxes:
                intersect_x1 = max(x1, bx1)
                intersect_y1 = max(y1, by1)
                intersect_x2 = min(x2, bx2)
                intersect_y2 = min(y2, by2)

                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    intersection_area += (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

            # Determine if the segment is clean or dirty
            if intersection_area / segment_area > overlap_threshold:
                clean_segments.append(False)
                color = (0, 0, 255)  # Red for dirty
            else:
                clean_segments.append(True)
                color = (0, 255, 0)  # Green for clean

            # Draw the segment rectangle on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Display the final image
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return clean_segments


# Inputs
# image_folder = "test/images/"
image_folder = "output_images/"

# bounding_box_folder = "test/labels/"
bounding_box_folder = "output_coordinates/"

overlap_threshold = 0.1 # Change this value as needed

# Iterate through all images in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(image_folder, image_file)
        
        # Find corresponding bounding box file
        boundin
g_box_file = os.path.join(bounding_box_folder, os.path.splitext(image_file)[0] + ".txt")
        if os.path.exists(bounding_box_file):
            print(f"Processing image: {image_path}")
            clean_segments = process_image(image_path, bounding_box_file, overlap_threshold)
            print(f"Clean segments for {image_file}: {clean_segments}")
        else:
            print(f"Bounding box file not found for image: {image_file}")



