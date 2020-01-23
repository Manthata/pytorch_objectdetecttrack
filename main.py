from models import *
import sys
sys.path.insert(1, './utils')
from utils import *
from torch.utils.data import DataLoader
import cv2
from sort import *
from coord import PixelMapper
import numpy as np
from detect import detect_image
import pandas as pd

#from world_coord import toworld
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) < (B[1]-A[1]) * (C[0]-A[0])

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size =416
conf_thres=0.8
nms_thres=0.4
classes = utils.load_classes(class_path)
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
quad_coords = {
#    "pixel": np.array([
        #[654, 263], #  top right
        #[419, 273], #top left
        #[241, 590], #  bottom left
        #[796, 595], # bottom right
        #]),

     "pixel": np.array([
            [531, 186], #  top right
            [186, 391], #top left
            [443, 599], #  bottom left
            [791, 314], # bottom right
    ]),
    "lonlat": np.array([
        [497, 145], #top right
        [177, 173], #top left
        [177, 437], # Bottom left
        [497, 421], # Bottom right


    ])

#    "lonlat": np.array([
#        [555, 100], #top right
#        [50, 100], #top left
#        [50, 705], # Bottom left
#        [555, 705], # Bottom right
#    ])
}

#floorP_pts = np.float32([
 #       [419, 273], #top left
  #      [654, 263], #  top right
   #     [241, 590], #  bottom left
    #    [796, 595], # bottom right
    #])

#frame_pts = np.float32([
 #       [73, 63], #top left
  #      [528, 63], #top right
   #     [73, 737], # Bottom left
    #    [528, 737], # Bottom right
    #])

vid = cv2.VideoCapture(0)
mot_tracker = Sort()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = vid.read()
#print(cap.shape)
#frame = cv2.resize(cap, (500, 500))
vw = frame.shape[1]
vh = frame.shape[0]
print(frame.shape)
print ("Video size", vh,vw)

frame_out = cv2.VideoWriter("outputs/frame_1.mp4",fourcc,20.0,(vw,vh))
floorplan_out = cv2.VideoWriter("outputs/floorplan_1.mp4",fourcc,20.0,(vw,vh))

all_lines = {}
all_lines_transformed = {}
frames = 0
starttime = time.time()
#floorplan = cv2.imread("newplan.png")
floorplan = cv2.imread("images/bedroom_plan.png")
floorplan = cv2.rotate(floorplan, cv2.ROTATE_90_COUNTERCLOCKWISE)

print("------------------------------ floor plan size", floorplan.shape)
#floorplan = cv2.imread("bedroom_floorplan.png")
floorplan = cv2.resize(floorplan, (600, 800))
print("---------------------------------", floorplan.shape )



#floorplan = cv2.rotate(floorplan, cv2.ROTATE_90_COUNTERCLOCKWISE)
#floorplan = np.zeros((600, 800, 3), np.uint8)
#floorplan = cv2.rectangle(floorplan, (0, 0), (800, 600), (255, 255, 255), -1)
#matrix = cv2.getPerspectiveTransform(frame_pts, floorP_pts)
#results = c
#cv2.warpPerspective(frame, matrix, (800, 600))
counter_up = 0
counter_down = 0
counter_left = 0
counter_right = 0
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
            #print("coordinates for mapped pixels=", pm)
          #  uv_0 = (0, 0) # Top left give way sign in frame
          #  lonlat_0 = pm.pixel_to_lonlat(uv_0)
          #  print(lonlat_0)


            #lonlat_1 = (35.388217, 139.425694) # Center of the roundabout on googlemaps
            #uv_1 = pm.lonlat_to_pixel(lonlat_1)
            #The line drawing part
            if not obj_id in all_lines:
                all_lines[obj_id] = []

            cord = all_lines[obj_id]
            # reduce the length of the line
            #if len(cord) > 50:
                #cord.pop(0)

            #x1 = int(x1 + box_w/2)
            #y1 = int(y1 +box_h/2)
            x1 = int(x1 +500)
            y1 = int(y1+500)
            cv2.circle(frame, (x1, y1), 5,color , -1)
            cord.append((x1,y1))
            if not obj_id in all_lines_transformed:
                all_lines_transformed[obj_id] = []

            cord_bbox = all_lines_transformed[obj_id]


            uv_bbox = (x1, y1) # The coordinates of the bounding box
            print("bbox coords = ",uv_bbox)
            lonlat_bbox = pm.pixel_to_lonlat(uv_bbox)
            uv_1 = pm.lonlat_to_pixel(lonlat_bbox)
            print("returned bbox_coords = ", uv_1)
            print("Longitudinal and latitudinal positions = ", lonlat_bbox)
            x1_transformed = lonlat_bbox[0][0]
            y1_transformed = lonlat_bbox[0][1]
            cord_bbox.append((int(x1_transformed), int(y1_transformed)))
            print(x1_transformed, y1_transformed)

            loop_count = 0
            for index, xp in enumerate(cord_bbox[:-1]):
                cv2.line(floorplan, xp, cord_bbox[index +1], color, 5, lineType = 8)
                print("This is our points:", xp, cord_bbox[index +1])

                cv2.line(floorplan,(176, 441),(496, 439), color,thickness=2, lineType=8, shift=0)
                cv2.line(floorplan,(363, 420),(364, 158), color,thickness=2, lineType=8, shift=0)
                y = [ i[1] for i in cord_bbox]
                x = [i[0] for i in cord_bbox]
                y_mean = np.mean(y)
                x_mean = np.mean(x)
                print("this is the mean of the y", y_mean)
                y_direction = cord_bbox[-1][1] - y_mean
                x_direction = cord_bbox[-1][0] - x_mean
                print("The direction:", x_direction, y_direction)
                loop_count += 1
            if loop_count > 1:
                if y_direction < 0 and intersect(cord_bbox[-1], cord_bbox[-2], (176, 441),(496, 439)):
                    counter_up += 1
                elif y_direction < 0 and intersect(cord_bbox[-1], cord_bbox[-2], (176, 441),(496, 439)):
                    counter_down += 1
                elif x_direction < 0 and intersect(cord_bbox[-1], cord_bbox[-2], (363, 420),(364, 158)):
                    counter_left += 1
                elif x_direction > 0 and intersect(cord_bbox[-1], cord_bbox[-2], (363, 420),(364, 158)):
                    counter_right += 1
            for index, xp in enumerate(cord[:-1]):
                cv2.line(frame, xp, cord[index +1], color, 5, lineType = 8)
            floorplan_out.write(floorplan)
            frame_out.write(frame)
           # for index, xp in enumerate(cord[:-1]):
          #     cv2.line(framew, xp, cord[index +1], color, 5, lineType = 8)


            #points on the input video

    cv2.circle(frame, (531, 186), 5, (0, 0, 255), -1)
    cv2.circle(frame, (186, 391), 5, (0, 0, 255), -1)
    cv2.circle(frame, (443, 599), 5, (0, 0, 255), -1)
    cv2.circle(frame, (791, 314), 5, (0, 0, 255), -1)
    #cv2.circle(frame, (241, 590), 5, (0, 0, 255), -1)
    #cv2.circle(frame, (796, 595), 5, (0, 0, 255), -1)
    pts = np.array([[531, 186],[186, 391],[443, 599],[791, 314]], np.int32)
   # [350, 397],[350, 397],[705, 397]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(frame,[pts],True,(0,0,255))
    #points of the floor plan
    cv2.circle(floorplan, (497, 145), 5, (0, 0, 255), -1)
    cv2.circle(floorplan, (177, 145), 5, (0, 0, 255), -1)
    cv2.circle(floorplan, (177, 437), 5, (0, 0, 255), -1)
    cv2.circle(floorplan, (497, 437), 5, (0, 0, 255), -1)
    pts = np.array([[497, 145],[177, 145],[177, 437],[497, 437]], np.int32)

    #Delta expand_dims
    #cv2.circle(floorplan, (50, 100), 5, (0, 0, 255), -1)
    #cv2.circle(floorplan, (555, 100), 5, (0, 0, 255), -1)
    #cv2.circle(floorplan, (50, 705), 5, (0, 0, 255), -1)
    #cv2.circle(floorplan, (555, 705), 5, (0, 0, 255), -1)
    #pts = np.array([[555, 100],[50, 100],[50, 705],[555, 705]], np.int32)

    pts = pts.reshape((-1,1,2))
    cv2.polylines(floorplan,[pts],True,(0,0,255))
    info = [
		("Up", counter_up),
		("Down", counter_down),
		("Left", counter_left),
        ("Right", counter_right)
	]

	# loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, 200 - ((i * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    #cv2.putText(frame, str(counter_up), (700,80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 255), 5)
    #cv2.putText(frame, str(counter_down), (500,80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 255), 5)
    #cv2.putText(frame, str(counter_left), (300,80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 255), 5)
    #cv2.putText(frame, str(counter_right), (100,80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 255), 5)
    df = pd.DataFrame({'Up': [counter_up],
                    'Down': [counter_down],
                    'Left':[counter_left],
                    'Right': [counter_right]})
    df.to_csv('csv_files/direction.csv')
    cv2.imshow('Stream_1', frame)
    cv2.imshow('Steam_2', floorplan)
    #cv2.imshow('Stream_3', results)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
floorplan_out.release()
frame_out.release()
