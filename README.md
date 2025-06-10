
# Flickd AI Hackathon



## 🗎 Initializing the YOLO Version 8(Mega) Model 

• Download the model file from the below link:
https://huggingface.co/Ultralytics/YOLOv8/resolve/8a9e1a55f987a77f9966c2ac3f80aa8aa37b3c1a/yolov8m.pt?download=true

• After downloading the model file, move it into the folder **\Flickd_AI_Hackathon-main** 

• Once the YOLOv8m is initialized, it can be accessed by the main.py.

## 🆔Entering the Video Id

The Video-Id has to be entered in the line-23 stored under the variable vid_id.

**Example of Video-Id: '2025-05-28_13-42-32_UTC'**

## 🖼️Auto saving the Cropped Images
During the time the main.py runs, the detected objects in the frame will be cropped and gets saved automatically in the directory **\Flickd_AI_Hackathon-main\runs\detect\predict**

## 🖹 Creation of JSON File
This is the final stage where a JSON file will be created with the file name **'final_js.json'** under the parent directory('\Flickd_AI_Hackathon-main\') itself.
