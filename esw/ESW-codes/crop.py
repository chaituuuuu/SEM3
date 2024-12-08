import os
from PIL import Image
import pytesseract
import re

# Path to the folder containing the images
input_folder = '/home/chaitu/Desktop/ESW pics/data'
output_folder = '/home/chaitu/Desktop/ESW pics/IMG2'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Custom configuration for OCR
custom_config = '--psm 12'

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust as needed for different formats
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)

        # Extract text from the image
        extracted_text = pytesseract.image_to_string(img, config=custom_config)

        # Use a regular expression to find a timestamp in the format dd-mm-yyyy hh:mm:ss
        match = re.search(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}", extracted_text)

        # Check if a match is found
        if match:
            # Extract the timestamp
            timestamp = match.group(0)
            
            # Format the timestamp to replace spaces and colons with valid filename characters
            formatted_timestamp = timestamp.replace(":", "-").replace(" ", "_")

            # Prepare the new filename based on the extracted timestamp
            new_image_name = f'{formatted_timestamp}.jpg'
            new_image_path = os.path.join(output_folder, new_image_name)

            # Save the image with the new name
            img.save(new_image_path)

            print(f"Image {filename} saved as: {new_image_name}")
        else:
            print(f"No valid timestamp found in the image: {filename}")

