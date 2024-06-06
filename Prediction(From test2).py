from ultralytics import YOLO
import torch
import cv2
import numpy as np
from datetime import datetime

model = YOLO('best.pt')

result = model.predict(source="ikan/code/ikan-30-4detik.mp4", show = True)
