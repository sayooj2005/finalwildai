import requests
import time
import itertools

url = 'http://localhost:8083/api/esp32/frame'
images = ['tiger_preview.jpg', 'elephant_img.png']
cycle_images = itertools.cycle(images)

print("Starting simulated ESP32-CAM stream...")
while True:
    img_path = next(cycle_images)
    try:
        with open(img_path, 'rb') as f:
            data = f.read()
        requests.post(url, data=data, timeout=1)
    except Exception as e:
        print(f"Failed to push {img_path}: {e}")
    time.sleep(0.5)
