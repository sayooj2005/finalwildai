import cv2
import requests
import time

# 1. Your Phone's IP Webcam URL
PHONE_CAMERA_URL = 'http://192.168.1.5:8080/video'

# 2. Your Render Cloud URL
CLOUD_API_URL = 'https://wild-ai-cloud-1.onrender.com/api/esp32/frame'

print("Connecting to Phone Camera at", PHONE_CAMERA_URL, "...")
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

if not cap.isOpened():
    print("ERROR: Could not connect to phone camera! Make sure IP Webcam app is running.")
    exit()

print("Connected! Streaming frames to Render Cloud...")
print("Open your Render Dashboard to view the AI processing!")

session = requests.Session()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from phone.")
        time.sleep(1)
        continue
        
    # Resize frame slightly to protect the Free Tier Cloud Memory
    frame = cv2.resize(frame, (640, 480))
        
    # Encode frame to JPEG
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not success:
        continue
        
    try:
        # Send to Render Cloud
        response = session.post(CLOUD_API_URL, data=buffer.tobytes(), headers={'Content-Type': 'image/jpeg'})
        if response.status_code == 200:
            print("Frame sent successfully to Cloud AI! (200 OK)")
        else:
            print(f"Cloud Error: {response.status_code}")
    except Exception as e:
        print(f"Failed to connect to cloud: {e}")
        
    # Wait 2 seconds between frames to prevent Cloudflare 429 Bans
    time.sleep(2)
