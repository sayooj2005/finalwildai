import cv2
import numpy as np
import time
from fastapi.testclient import TestClient
from wildlife_intrusion_api import app

def test_websocket_stream_and_frame_ingestion():
    print("Testing FastAPI app startup and WebSocket streaming...")
    with TestClient(app) as client:
        # Create a test frame (dummy RGB image with an elephant/tiger or blank)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "WildAI Stream Test", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        frame_bytes = buffer.tobytes()

        with client.websocket_connect("/ws/client") as websocket:
            print("Connected to /ws/client WebSocket successfully.")
            
            # First message on connect is camera_status handshake
            init_msg = websocket.receive_json()
            print("Received initial handshake message:", init_msg)
            assert init_msg.get("type") == "camera_status"
            assert init_msg.get("status") == "connected"

            # Post test frame to /api/esp32/frame
            response = client.post("/api/esp32/frame", content=frame_bytes, headers={"Content-Type": "image/jpeg"})
            assert response.status_code == 200, f"Frame post failed: {response.text}"
            print("Posted frame to /api/esp32/frame successfully.")

            # Receive detection payload from WebSocket (loop until detection received)
            data = None
            for _ in range(5):
                msg = websocket.receive_json()
                print("Received WebSocket payload type:", msg.get("type"))
                if msg.get("type") == "detection":
                    data = msg
                    break

            assert data is not None, "Did not receive detection payload"
            assert data.get("type") == "detection", f"Expected detection payload, got {data}"
            assert "processed_frame" in data, "Missing processed_frame in payload"
            assert data["processed_frame"].startswith("data:image/jpeg;base64,"), "Invalid processed_frame data URI format"
            assert "diagnostics" in data, "Missing diagnostics"
            print("WebSocket transmission verified! Diagnostics:", data.get("diagnostics"))

if __name__ == "__main__":
    test_websocket_stream_and_frame_ingestion()
    print("ALL STREAM VALIDATION CHECKS PASSED!")
