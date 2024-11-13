import random
import cv2
import numpy as np
from ultralytics import YOLO
import os
try:
    import winsound
except ImportError:
    print("Winsound Not found")

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()
WHITE = (255, 255, 255)
focal_length_found = 680.0
KNOWN_WIDTH = 8
# print(class_list)
fl={"person":550,"bicycle":800,"car":750,"traffic light":400,"fire hydrant":900,"stop sign":800,"backpack":860,"tie":680,"bottle":680,"wine glass":680,"cup":680,"chair":860,"laptop":800,"keyboard":800,"cell phone":680,"book":800,"clock":800}
aw={"person":40,"bicycle":100,"car":176,"traffic light":30,"fire hydrant":15,"stop sign":30,"backpack":35,"tie":8,"bottle":8,"wine glass":8,"cup":8,"chair":42,"laptop":30,"keyboard":30,"cell phone":8,"book":30,"clock":30}
# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("inference/videos/tikkinebanayi.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# face detector function
def object_data(image,faces):

    object_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), WHITE, 1)
        face_width = w

    return face_width

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    i-=1
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            
            text_bottom_right = (frame.shape[1] - 10, frame.shape[0] - 10)
            text_bottom_left = (10, frame.shape[0] - 10)
            font_scale = 1
            font_thickness = 2
            font = cv2.FONT_HERSHEY_COMPLEX
            # cv2.putText(frame, ("Distance" + str(Distance)), text_bottom_right, font, font_scale, (255, 255, 255), font_thickness)
            


            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            object_width_in_frame = object_data(frame,box.xywh.numpy())
            if object_width_in_frame != 0:
                try:
                    Distance = distance_finder(fl[class_list[int(clsID)]],aw[class_list[int(clsID)]],object_width_in_frame)
                    if Distance < 50:
                        try:
                            winsound.Beep(500,1000)
                        except Exception as e:
                            os.system('beep -f %s -l %s' % (500,1000))
                except Exception as e:
                    Distance = distance_finder(750,30,object_width_in_frame)

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + "Distance" + str(round(Distance,3)),
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
