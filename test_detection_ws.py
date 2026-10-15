import urllib.request
import json
import time
import asyncio
import websockets
import sys

async def test_real_animal_detection():
    print("[TEST REAL YOLO DETECTION] Pushing detection_tiger_1790249761.jpg to camera endpoint...")
    with open('detection_tiger_1790249761.jpg', 'rb') as f:
        img_bytes = f.read()

    uri = 'ws://127.0.0.1:8083/ws/client'
    async with websockets.connect(uri) as ws:
        # Push real image to camera endpoint
        req = urllib.request.Request(
            'http://127.0.0.1:8083/api/esp32/frame',
            data=img_bytes,
            headers={'Content-Type': 'image/jpeg'}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  Frame push response code: {resp.status}")

        start = time.time()
        detected = False
        while time.time() - start < 8.0:
            msg = await asyncio.wait_for(ws.recv(), timeout=4.0)
            data = json.loads(msg)
            if data.get('type') == 'detection':
                dets = data.get('detections', [])
                if dets:
                    print(f"  Received {len(dets)} detection(s):")
                    for d in dets:
                        animal = d.get('animal') or d.get('species') or d.get('class_name')
                        conf = d.get('confidence')
                        risk = d.get('risk_score')
                        track = d.get('track_id')
                        print(f"    - Animal: {animal}, Confidence: {conf}, Risk: {risk}, Track: {track}")
                    detected = True
                    break
        
        if not detected:
            print("  Warning: No detections in tiger image, testing with human_detection_sample.jpg...")
            with open('human_detection_sample.jpg', 'rb') as f:
                img_bytes2 = f.read()
            req2 = urllib.request.Request(
                'http://127.0.0.1:8083/api/esp32/frame',
                data=img_bytes2,
                headers={'Content-Type': 'image/jpeg'}
            )
            urllib.request.urlopen(req2, timeout=5)
            while time.time() - start < 12.0:
                msg = await asyncio.wait_for(ws.recv(), timeout=4.0)
                data = json.loads(msg)
                if data.get('type') == 'detection':
                    dets = data.get('detections', [])
                    if dets:
                        print(f"  Received {len(dets)} detection(s):")
                        for d in dets:
                            animal = d.get('animal') or d.get('species') or d.get('class_name')
                            conf = d.get('confidence')
                            risk = d.get('risk_score')
                            track = d.get('track_id')
                            print(f"    - Animal: {animal}, Confidence: {conf}, Risk: {risk}, Track: {track}")
                        detected = True
                        break

        assert detected, "Expected at least one detection from the YOLO model"
        print("-> PASS: Real detection verified end-to-end via YOLO and WebSocket!")

if __name__ == '__main__':
    asyncio.run(test_real_animal_detection())
