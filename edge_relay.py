"""
WildAI Edge Camera Relay Agent
----------------------------------------------------------------------
Bridges local private IP cameras (192.168.x.x) or USB/RTSP streams to the
Cloud Backend (Render / Railway) over outbound HTTPS.

Key Features:
- Zero firewall or router port-forwarding needed (outbound HTTPS).
- Solves Mixed Content: Web browsers on Vercel (HTTPS) can view local cameras.
- Auto-reconnects with exponential backoff on stream interruptions.
- Adapts frame size to 640x480 for smooth, low-latency transmission.

Usage:
    python edge_relay.py --source http://192.168.1.5:8080/video --cloud https://finalwildai.onrender.com
    python edge_relay.py --source 0 --cloud https://finalwildai.onrender.com
"""

import argparse
import sys
import time
import cv2

try:
    import requests
except ImportError:
    print("[!] 'requests' library required. Install with: pip install requests")
    sys.exit(1)

def run_edge_relay(source, cloud_url, fps_limit=20, quality=65, camera_id="EDGE_RELAY_01"):
    push_endpoint = cloud_url.rstrip("/") + "/api/camera/push_frame"
    
    # Notify cloud backend that this camera stream is being ingested via push
    try:
        settings_url = cloud_url.rstrip("/") + "/api/settings/camera"
        requests.post(settings_url, json={"url": "PUSH_INGEST"}, timeout=5)
        print(f"[✓] Notified cloud pipeline: set to PUSH_INGEST mode.")
    except Exception as e:
        print(f"[!] Warning: Could not notify cloud settings ({e}), continuing push directly...")

    print("==================================================================")
    print(" WildAI Edge Surveillance Relay")
    print(f" Source Camera  : {source}")
    print(f" Cloud Endpoint : {push_endpoint}")
    print(f" Target FPS     : {fps_limit} | JPEG Quality: {quality}%")
    print(" Press Ctrl+C to stop.")
    print("==================================================================")

    session = requests.Session()
    frame_delay = 1.0 / float(fps_limit) if fps_limit > 0 else 0.05
    src_input = int(source) if str(source).isdigit() else str(source)

    while True:
        print(f"[*] Opening camera source: {source} ...")
        cap = cv2.VideoCapture(src_input)
        if not cap.isOpened():
            print(f"[!] Unable to open camera source: {source}. Retrying in 3 seconds...")
            time.sleep(3)
            continue

        print(f"[✓] Camera connected! Streaming frames to Cloud AI...")
        consecutive_errors = 0
        frames_sent = 0
        t_start = time.time()

        try:
            while cap.isOpened():
                t_frame_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 25:
                        print("[!] Stream interrupted. Reconnecting...")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_errors = 0

                # Downscale for optimal bandwidth over cloud connections
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

                # JPEG encode
                _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                frame_bytes = buf.tobytes()

                # Push to Cloud Backend
                try:
                    resp = session.post(
                        push_endpoint,
                        data=frame_bytes,
                        headers={
                            'Content-Type': 'image/jpeg',
                            'X-Camera-ID': camera_id
                        },
                        timeout=1.5
                    )
                    if resp.status_code == 200:
                        frames_sent += 1
                        if frames_sent % 100 == 0:
                            elapsed = time.time() - t_start
                            actual_fps = frames_sent / elapsed if elapsed > 0 else 0
                            print(f"[RELAY] Transmitted {frames_sent} frames ({actual_fps:.1f} FPS)")
                except Exception as ex:
                    # Drop frame on network jitter
                    pass

                # Rate limiting
                elapsed_frame = time.time() - t_frame_start
                sleep_time = frame_delay - elapsed_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[*] Stopping relay gracefully...")
            break
        except Exception as e:
            print(f"[!] Relay loop exception: {e}")
        finally:
            cap.release()
            print("[*] Camera capture released.")

        time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="WildAI Local IP Camera to Cloud Relay")
    parser.add_argument("--source", "-s", default="http://192.168.1.5:8080/video",
                        help="Camera source: IP webcam URL (e.g. http://192.168.1.5:8080/video) or local index (0)")
    parser.add_argument("--cloud", "-c", default="https://finalwildai.onrender.com",
                        help="Cloud backend base URL (e.g. https://finalwildai.onrender.com or http://localhost:8083)")
    parser.add_argument("--fps", type=int, default=18, help="Target frame rate (default: 18 FPS)")
    parser.add_argument("--quality", type=int, default=65, help="JPEG compression quality (1-100, default: 65)")
    parser.add_argument("--id", default="EDGE_RELAY_01", help="Identifier for this camera feed")

    args = parser.parse_args()
    run_edge_relay(
        source=args.source,
        cloud_url=args.cloud,
        fps_limit=args.fps,
        quality=args.quality,
        camera_id=args.id
    )

if __name__ == "__main__":
    main()
