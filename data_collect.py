import cv2
import os
import shutil
import torch
from simplemodel import SimpleModel

from torchvision import transforms

save_folder = "./data"
model_folder = "./models"

data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = SimpleModel()
model.load_state_dict(torch.load(model_folder + "/traditional.pt"))


if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

def clear_data():
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    os.mkdir(save_folder + "/0")
    os.mkdir(save_folder + "/1")
    os.mkdir(save_folder + "/2")

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
    os.mkdir(save_folder + "/0")
    os.mkdir(save_folder + "/1")
    os.mkdir(save_folder + "/2")

    

FULL_CAPTURE_WIN = "Full_capture"
CROP_WIN = "Crop"
COLOR_CROP = (0, 0, 255)

RECORD_COLOR = [(0, 255, 0), (0, 0, 255)]
REC_POSITION = (50, 80)
REC_RADIUS = 20
LABEL_POSITION = (5, 25)
FONT = cv2.FONT_HERSHEY_TRIPLEX
FONT_SIZE = 0.6
FONT_CLR = 255

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.namedWindow(CROP_WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(CROP_WIN, 800, 600)

current_frame = 0
current_label = 0


imgs_collected = {0:len(os.listdir(save_folder + "/0")), 1:len(os.listdir(save_folder + "/1")), 2:len(os.listdir(save_folder + "/2"))}
is_recording = False

while ret:
    ret, frame = cap.read()
    new_frame = frame.copy()
    cv2.putText(new_frame, f'Label: {current_label} with {imgs_collected[current_label]} pics', (10, 430), FONT, 1, (255,255,255))
    cv2.imshow(CROP_WIN, new_frame)

    resized = cv2.resize(frame, (224, 224))
    if is_recording:
        imgs_collected[current_label] += 1
        
        cv2.imwrite(save_folder + f"/{current_label}/{imgs_collected[current_label]}.png", resized)
    key = cv2.waitKey(1)
    if chr(key % 256) == "0":
        current_label = 0
    elif chr(key % 256) == "1":
        current_label = 1
    elif chr(key % 256) == "2":
        current_label = 2
    elif chr(key % 256) == "q":
        break
    elif chr(key % 256) == "r":
        is_recording = not is_recording
    elif chr(key % 256) == "n":
        clear_data()
        imgs_collected[0] = 0
        imgs_collected[1] = 0
        imgs_collected[2] = 0
        is_recording = False