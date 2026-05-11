from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="Fish Counter", job_type="training")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8n"
dataset_name = "config2.yaml"
model = YOLO(f"{model_name}.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
model.train(project="Fish Counter", data=dataset_name, epochs=250, imgsz=640)

# Step 5: Validate the Model
model.val()

# Step 6: Perform Inference and Log Results
model(["path/to/image1", "path/to/image2"])

# Step 7: Finalize the W&B Run
wandb.finish()