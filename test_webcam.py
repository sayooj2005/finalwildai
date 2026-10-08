import cv2
import requests
import time

API_URL = 'http://localhost:8083/api/esp32/frame'

print("Initializing Laptop Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Streaming live webcam frames to {API_URL}...")
print("Open http://localhost:8083/admin-portal.html to view the AI processing!")
print("Press Ctrl+C to stop.")

# Create a persistent HTTP session to remove TCP handshake overhead on every frame!
session = requests.Session()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Encode frame as JPEG (High Quality)
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue
            
        # POST the raw bytes using the persistent session
        try:
            session.post(API_URL, data=buffer.tobytes(), timeout=1)
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}")
            time.sleep(0.5)
            
        # Absolute maximum hardware FPS - No sleep limits
        
except KeyboardInterrupt:
    print("\nStopping webcam stream.")
finally:
    cap.release()
