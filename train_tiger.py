from ultralytics import YOLO, settings
import os

# Fix permission error by setting dataset directory to the local folder
settings.update({'datasets_dir': os.path.join(os.getcwd(), 'datasets')})

print("=== WildAI Tiger Training Script ===")
print("Starting YOLO11 custom training process for Tiger Detection.")
print("Warning: If you do not have an NVIDIA GPU, this may take a long time on CPU.")

# Load the new YOLO11 Nano model we just downloaded
model = YOLO('yolo11n.pt')

# Train the model on the Tiger dataset
# Note: 'tiger-pose.yaml' is an official Ultralytics dataset that will automatically download!
results = model.train(
    data='tiger-pose.yaml',
    epochs=5,         # 5 loops over the data (Keep it low so it doesn't take 10 hours)
    imgsz=320,        # Keep image size small for speed and low RAM
    batch=4,          # Small batch size for laptops
    name='tiger_model'
)

print("\n=== Training Complete! ===")
print("Your custom Tiger AI brain is saved in the 'runs/detect/tiger_model/weights/' folder as 'best.pt'!")
