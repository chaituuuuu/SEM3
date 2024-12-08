# Google Collab Link

https://colab.research.google.com/drive/14RSCDeUXQh_yjWJOEK98W60L4Nsj2EvZ?usp=sharing

# API key for roboflow model

```
#import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="u3NpMO9V1kTHv2REIkTi"
)

# infer on a local image
result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="final-final-3/1")

```

# video_img_cord.py

- converting video into the frames and then their bounding box co-ordinates . this will be saved in 2 different folders . These folders will be further used in ROI.py

# ROI.py

- this code will display the images in matlab and will display red or green based on dirty and clean. 

