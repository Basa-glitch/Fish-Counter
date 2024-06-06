from ultralytics import YOLO

#Load a model
model = YOLO("yolov8n.yaml") #Build a model from scratch

#Use the model
results = model.train(data="config.yaml", epochs=200) #Train the model