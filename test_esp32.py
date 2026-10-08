import requests
import time

url = 'http://localhost:8083/api/esp32/frame'
try:
    with open('tiger_preview.jpg', 'rb') as f:
        data = f.read()
    
    print("Sending frame to", url)
    r = requests.post(url, data=data)
    print("Response:", r.status_code, r.text)
except Exception as e:
    print("Error:", e)
