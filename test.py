import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import pandas as pd
from collections import Counter
import glob

model = YOLO("yolov8s.pt")

my_file = open("Fish.txt", "r")
data = my_file.read()
class_list = data.split("\n")

def objects(img):
    results = model.predict(img)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    object_classes = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        # Check the structure of the model's output
        print("Model Output:", row)

        # Set the label directly to "fish"
        obj_class = "fish"
        object_classes.append(obj_class)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(img, f'{obj_class}', (x2, y2), 1, 1)

    return object_classes

def count_objects_in_image(object_classes):
    counter = Counter(object_classes)
    print("Object Count in Image:")
    for obj, count in counter.items():
        print(f"{obj}: {count}")

path = "C:/Users/bryan/OneDrive/Desktop/Coding/Python/FishCounter/Images/*"
for file in glob.glob(path):
    img = cv2.imread(file)
    img = cv2.resize(img, (1020, 500))
    object_classes = objects(img)
    count_objects_in_image(object_classes)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()