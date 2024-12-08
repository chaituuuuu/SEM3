import cv2
import os

# Playing video from file:
cap = cv2.VideoCapture('v4.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# Calculate the interval to save 5 frames per second
frame_interval = int(fps / 1)

currentFrame = 0
savedFrame = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break  # Exit the loop if no frame is captured

    # Save every 'frame_interval' frame
    if currentFrame % frame_interval == 0:
        name = './data/frame' + str(savedFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)
        savedFrame += 1

    # Increment the frame counter
    currentFrame += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()