from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to(device)
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

#videopath = 'live_traffic.mp4'

import cv2
from sort import *
#from willima_transform import Pespective_transform
from coord import PixelMapper
import numpy as np
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
quad_coords = {
    "lonlat": np.array([
        [544, 60], #  top right
        [69, 61], #top left
        [69, 744], #  bottom left
        [544, 744], # bottom right
    ]),
    "pixel": np.array([
        [800, 0], #top right
        [0, 0], #top left
        [0, 600], # Bottom left
        [800, 600], # Bottom right
    ])
}

vid = cv2.VideoCapture(0)
mot_tracker = Sort() 


#fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, cap = vid.read()
frame = cv2.resize(cap, (610, 808))
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)

#outvideo = cv2.VideoWriter("output_1.mp4",fourcc,20.0,(vw,vh))
all_lines = {}
all_lines_transformed = {}
frames = 0
starttime = time.time()
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    

          #  warped = four_point_transform(frame, pts)
            pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])
          #  uv_0 = (0, 0) # Top left give way sign in frame
          #  lonlat_0 = pm.pixel_to_lonlat(uv_0)
          #  print(lonlat_0)
            
            #matrix = cv2.getPerspectiveTransform(frame_pts, floorP_pts)
            #result = cv2.warpPerspective(frame, matrix, (1280, 720))
            #lonlat_1 = (35.388217, 139.425694) # Center of the roundabout on googlemaps
            #uv_1 = pm.lonlat_to_pixel(lonlat_1)
            #The line drawing part 
            if not obj_id in all_lines:
                all_lines[obj_id] = []
                
            cord = all_lines[obj_id]
            # reduce the length of the line 
            #if len(cord) > 50:
                #cord.pop(0)
            x1 = int(x1 + box_w/2)
            y1 = int(y1 +box_h/2)
            cord.append((x1,y1))
            output = cv2.imread("bedroom_plan.png")
            print("---------------------------------", output.shape)
            Rotated_output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
          
            print(Rotated_output.size)
            
            if not obj_id in all_lines_transformed:
                all_lines_transformed[obj_id] = []
                
            cord_bbox = all_lines_transformed[obj_id]
            
            uv_bbox = (x1, y1) # The coordinates of the bounding box
            lonlat_bbox = pm.pixel_to_lonlat(uv_bbox)
            print(lonlat_bbox)
            x1_transformed = lonlat_bbox[0][0]
            y1_transformed = lonlat_bbox[0][1]
            cord_bbox.append((int(x1_transformed), int(y1_transformed)))
            print(x1_transformed, y1_transformed)
           
            for index, xp in enumerate(cord_bbox[:-1]):
                 cv2.line(Rotated_output, xp, cord_bbox[index +1], color, 5, lineType = 8)
                
            for index, xp in enumerate(cord[:-1]):
                cv2.line(frame, xp, cord[index +1], color, 5, lineType = 8)
                
           # for index, xp in enumerate(cord[:-1]):
           #     cv2.line(framew, xp, cord[index +1], color, 5, lineType = 8)
                
                
                
                
            cv2.circle(frame, (100, 150), 5, (0, 0, 255), -1)
            cv2.circle(frame, (0, 500), 5, (0, 0, 255), -1)
            cv2.circle(frame, (600, 600), 5, (0, 0, 255), -1)
            cv2.circle(frame, (790, 150), 5, (0, 0, 255), -1)
            pts = np.array([[100, 150],[0, 500],[600, 600],[790, 150]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame,[pts],True,(0,0,255))
           
            
    cv2.imshow('Stream_1', frame)
    cv2.imshow('Steam_2', Rotated_output)
    #outvideo.write(output)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
#outvideo.release()
