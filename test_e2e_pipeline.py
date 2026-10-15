import urllib.request
import json
import time
import sys
import numpy as np
import cv2

def test_camera_status():
    print("[TEST 1] Testing GET /api/camera/status...")
    req = urllib.request.Request("http://127.0.0.1:8083/api/camera/status")
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200, f"Expected 200, got {response.status}"
        data = json.loads(response.read().decode('utf-8'))
        print(f"  Status response: {data}")
        assert "camera_status" in data, "Missing camera_status"
        print("  -> PASS: Camera status endpoint working properly.")

def test_push_frame_and_mjpeg():
    print("[TEST 2] Pushing synthetic test frame to /api/esp32/frame...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw green canvas and text
    cv2.putText(img, "WILDLIFE TEST FRAME", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    _, buf = cv2.imencode('.jpg', img)
    jpg_bytes = buf.tobytes()

    req = urllib.request.Request(
        "http://127.0.0.1:8083/api/esp32/frame",
        data=jpg_bytes,
        headers={"Content-Type": "image/jpeg"}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200, f"Expected 200, got {response.status}"
        res = json.loads(response.read().decode('utf-8'))
        print(f"  Frame push response: {res}")
        assert res.get("status") in ["ok", "success"]
    
    print("  -> Waiting 0.5s for inference pipeline...")
    time.sleep(0.5)

    print("[TEST 3] Testing GET /video_feed (MJPEG streaming)...")
    req = urllib.request.Request("http://127.0.0.1:8083/video_feed")
    with urllib.request.urlopen(req, timeout=5) as response:
        content_type = response.headers.get("Content-Type", "")
        print(f"  Content-Type: {content_type}")
        assert "multipart/x-mixed-replace" in content_type, f"Unexpected Content-Type: {content_type}"
        
        # Read first chunk
        chunk = response.read(8192)
        assert b"--frame" in chunk, "Expected --frame boundary in stream"
        assert b"\xff\xd8" in chunk, "Expected JPEG SOI marker (\xff\xd8) in stream"
        print(f"  Received {len(chunk)} bytes containing valid MJPEG frame boundaries and JPEG bytes.")
        print("  -> PASS: Video feed endpoint streams valid multipart MJPEG frames.")

def test_websocket():
    print("[TEST 4] Testing WebSocket /ws/client telemetry...")
    import asyncio
    import websockets

    async def run_ws():
        uri = "ws://127.0.0.1:8083/ws/client"
        async with websockets.connect(uri) as ws:
            # Send another frame to trigger broadcast
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "WS TELEMETRY TEST", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            _, buf = cv2.imencode('.jpg', img)
            req = urllib.request.Request("http://127.0.0.1:8083/api/esp32/frame", data=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
            urllib.request.urlopen(req, timeout=5)

            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            print(f"  Received WS message type: {data.get('type')}")
            assert "type" in data, "WS payload missing type"
            if data.get("type") == "detection":
                print(f"  Detections array present: {type(data.get('detections'))}")
                print(f"  Diagnostics: {data.get('diagnostics')}")
            print("  -> PASS: WebSocket client telemetry stream working properly.")

    asyncio.run(run_ws())

if __name__ == "__main__":
    try:
        test_camera_status()
        test_push_frame_and_mjpeg()
        test_websocket()
        print("\n=== ALL PIPELINE TESTS PASSED SUCCESSFULLY! ===")
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
