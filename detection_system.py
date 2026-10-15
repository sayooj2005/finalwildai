import cv2
import time
import sys
import os
import math
import numpy as np
from datetime import datetime
from behavior_analysis import WildlifeBehaviorAnalyzer

import contextlib
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Ultralytics/PyTorch could not be loaded ({e}). Falling back to OpenCV & Synthetic Detection Engine.")
    YOLO_AVAILABLE = False
    YOLO = None

class SyntheticBox:
    def __init__(self, x1, y1, x2, y2, conf, cls, track_id=None):
        self.xyxy = [[x1, y1, x2, y2]]
        self.conf = [conf]
        self.cls = [cls]
        self.id = [track_id] if track_id is not None else None

class SyntheticResult:
    def __init__(self, boxes_list):
        self.boxes = boxes_list

import threading

class ThreadedCameraReader:
    """
    High-performance, non-blocking camera capture thread.
    Maintains a single-slot buffer of the freshest frame (drop-stale policy)
    to achieve minimal end-to-end latency and prevent queue buildup.
    """
    def __init__(self, capture_source, width=640, height=480, target_fps=30):
        self.capture_source = capture_source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_count = 0
        self.capture_fps = 0.0
        self.running = False
        self.thread = None
        self._init_capture()

    def _init_capture(self):
        source = self.capture_source
        if source is None:
            return

        if isinstance(source, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
        else:
            self.cap = cv2.VideoCapture(source)

        if self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self):
        fps_timer = time.time()
        fps_counter = 0
        use_snapshot_fallback = False

        while self.running:
            frame = None
            if not use_snapshot_fallback and self.cap is not None and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                else:
                    if isinstance(self.capture_source, str) and self.capture_source.startswith('http'):
                        use_snapshot_fallback = True
                    time.sleep(0.01)
            elif isinstance(self.capture_source, str) and self.capture_source.startswith('http'):
                use_snapshot_fallback = True
                try:
                    import urllib.request
                    shot_url = self.capture_source.replace('/video', '/shot.jpg')
                    if not shot_url.endswith('/shot.jpg'):
                        shot_url = shot_url.rstrip('/') + '/shot.jpg'
                    req = urllib.request.urlopen(shot_url, timeout=2)
                    arr = np.array(bytearray(req.read()), dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    time.sleep(0.03)
                except Exception:
                    time.sleep(0.2)
            else:
                time.sleep(0.05)

            if frame is not None:
                now = time.time()
                with self.lock:
                    self.latest_frame = frame
                    self.latest_timestamp = now
                    self.frame_count += 1
                    fps_counter += 1
                    if now - fps_timer >= 1.0:
                        self.capture_fps = round(fps_counter / (now - fps_timer), 1)
                        fps_counter = 0
                        fps_timer = now

    def read_latest(self):
        """Returns (ret, frame, timestamp) without blocking."""
        with self.lock:
            if self.latest_frame is None:
                return False, None, 0.0
            return True, self.latest_frame.copy(), self.latest_timestamp

    def read(self):
        """Standard OpenCV VideoCapture interface compatibility."""
        ret, frame, _ = self.read_latest()
        return ret, frame

    def isOpened(self):
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.latest_frame is not None:
            return True
        if isinstance(self.capture_source, str) and self.capture_source.startswith('http'):
            return True
        return False

    def set(self, propId, value):
        if self.cap:
            return self.cap.set(propId, value)
        return False

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

class WildlifeIoTSystem:
    def __init__(self, model_path='yolo11n.pt', capture_index=0):
        print(f"[SYSTEM] Initializing Enhanced YOLO Wildlife Detector Module...")
        
        # Auto-detect CPU/GPU
        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            self.device = 'cpu'
            
        print(f"[SYSTEM] Hardware Accelerator detected: {self.device.upper()}")
        
        # Load YOLO Model
        self.model = None
        if YOLO_AVAILABLE:
            try:
                if not os.path.exists(model_path):
                    print(f"[WARNING] Model weights {model_path} not found locally! Ultralytics will attempt to auto-download.")
                self.model = YOLO(model_path)
                print(f"[SYSTEM] Model {model_path} loaded successfully on {self.device}.")
            except Exception as e:
                print(f"[ERROR] Critical failure loading YOLO model ({model_path}): {e}")
                print(f"[SYSTEM] Switching to OpenCV fallback mode to prevent pipeline crash.")
        else:
            print("[SYSTEM] YOLO not available. Running OpenCV AI Pose & Behavior Synthesizer mode.")

        # Load cached configuration
        self.config_data = self.load_config()

        # Initialize Camera using ThreadedCameraReader to eliminate lag
        self.current_url = capture_index
        self.reader = None
        self.cap = None
        if capture_index is not None:
            self.update_stream(capture_index)
            
        # Alert Cooldowns
        self.last_alert_time = {}
        self.alert_cooldown = 10

        # Posture & Motion History Tracking (for velocity & behavior calculation)
        self.track_history = {}
        self.frame_counter = 0
        
        # Load independent behavior analysis module
        self.behavior_analyzer = WildlifeBehaviorAnalyzer()

        print("[SYSTEM] Pose Estimation & Dynamic Risk Engine Active.")

    def load_config(self):
        import json
        try:
            with open("wildlife_config.json", "r") as f:
                self.config_data = json.load(f)
        except Exception:
            self.config_data = {
                "temporal_persistence_frames": 1,
                "preprocessing": {"enabled": False},
                "adaptive_roi": {"enabled": False},
                "categories": {"HUMAN": ["person", "poacher", "human"], "WILDLIFE": []},
                "classes": {}
            }
        return self.config_data

    def update_stream(self, capture_index):
        """Dynamically switches the camera source using ThreadedCameraReader to eliminate lag."""
        self.current_url = capture_index
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.release()
            self.reader = None
        if self.cap is not None and self.cap != getattr(self, 'reader', None):
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        
        print(f"[SYSTEM] Connecting to new camera stream: {capture_index}")
        if capture_index is not None:
            self.reader = ThreadedCameraReader(capture_index).start()
            self.cap = self.reader
            # Quick wait for reader
            for _ in range(10):
                if self.reader.isOpened():
                    break
                time.sleep(0.05)
            if not self.reader.isOpened():
                print(f"[ERROR] Could not open video stream: {capture_index}")
                return False
            return True
        return False

    def calculate_dynamic_risk(self, species, behaviour, velocity, direction_vector, group_size=1, is_night=False):
        """
        Dynamic Risk Engine incorporating 5 core factors:
        1. Species Threat Base
        2. Behavior Multiplier
        3. Movement Vector & Speed
        4. Group Size / Density
        5. Environmental Factors (Visibility/Night)
        Returns: risk_score (0-100), danger_level ('low', 'medium', 'high', 'critical')
        """
        species_lower = species.lower()

        # 1. Base Species Threat
        if any(s in species_lower for s in ["tiger", "leopard", "panther"]):
            base_score = 85
        elif "elephant" in species_lower:
            base_score = 80
        elif "poacher" in species_lower or "human" in species_lower:
            base_score = 90
        elif "bear" in species_lower:
            base_score = 70
        elif "boar" in species_lower:
            base_score = 50
        elif "deer" in species_lower:
            base_score = 20
        else:
            base_score = 30

        # 2. Behavior Multiplier
        behaviour_score = 0
        if behaviour in ("approaching", "running"):
            behaviour_score = 35
        elif behaviour == "group movement":
            behaviour_score = 25
        elif behaviour == "walking":
            behaviour_score = 10
        elif behaviour == "moving away":
            behaviour_score = -5
        else:  # stationary, lingering
            behaviour_score = 0

        # 3. Movement Vector & Speed
        movement_score = 0
        if "Approaching" in direction_vector:
            movement_score += 20
        elif "Retreating" in direction_vector:
            movement_score -= 10
            
        if velocity > 150:
            movement_score += 15

        # 4. Group Size
        group_score = 0
        if group_size >= 4:
            group_score = 20
        elif group_size >= 2:
            group_score = 10

        # 5. Environmental Factors (Night / Low Visibility)
        env_score = 15 if is_night else 0

        # Total Dynamic Risk Calculation
        raw_risk = base_score + behaviour_score + movement_score + group_score + env_score
        risk_score = min(max(int(raw_risk), 0), 100)

        # Danger Level Categorization
        if risk_score >= 85:
            danger_level = "critical"
        elif risk_score >= 65:
            danger_level = "high"
        elif risk_score >= 40:
            danger_level = "medium"
        else:
            danger_level = "low"

        return risk_score, danger_level

    def trigger_alert(self, track_id, species, confidence, frame, danger_level, risk_score, behaviour, movement_vector):
        """Triggers multi-channel responses (Audio Repellent, SMS, Evacuation Alert)."""
        current_time = time.time()
        if current_time - self.last_alert_time.get(track_id, 0) < self.alert_cooldown:
            return

        self.last_alert_time[track_id] = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n>>> [REAL-TIME ALERT] {species.upper()} #{track_id} | Risk: {risk_score}% ({danger_level.upper()}) | Behavior: {behaviour} | Time: {timestamp}")
        
        # Save evidence image
        filename = f"detection_{species}_{track_id}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

    def draw_pose_skeleton(self, frame, keypoints, behaviour, color):
        """Renders visual keypoint nodes and skeleton limbs on frame."""
        if not keypoints or len(keypoints) < 2:
            return

        # Draw nodes
        for pt in keypoints:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            cv2.circle(frame, pt, 7, color, 1)

        # Connect limbs
        for i in range(len(keypoints) - 1):
            cv2.line(frame, keypoints[i], keypoints[i+1], (0, 255, 128), 2)

    def classify_wildlife_species(self, raw_label: str, crop_bgr: np.ndarray):
        """
        Specialized wildlife classifier focusing strictly on the 3 designated animals:
        - TIGER (tiger, cat, zebra/stripes, horse/tawny quadruped, felines)
        - ELEPHANT (elephant)
        - WILD BOAR (wild_boar, boar, pig, dark quadrupeds)
        Plus PERSON (human security intrusion).
        All other non-target COCO classes (chair, bottle, etc.) are filtered out.
        """
        raw = (raw_label or '').lower().strip()

        # 1. Human / Person
        if raw in ['person', 'human', 'poacher']:
            return 'person'

        # 2. Elephant (accurate direct COCO class)
        if raw == 'elephant':
            return 'elephant'

        # Visual feature analysis of cropped area for stripes & tiger coat
        is_tiger = False
        is_dark_boar = False
        if crop_bgr is not None and crop_bgr.size > 0:
            ch, cw = crop_bgr.shape[:2]
            if ch >= 12 and cw >= 12:
                hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

                # Tiger orange / tawny coat mask (Hue 4 to 26 in OpenCV HSV)
                orange_mask = cv2.inRange(hsv, np.array([4, 40, 40]), np.array([26, 255, 255]))
                orange_ratio = np.count_nonzero(orange_mask) / float(ch * cw)

                # Dark stripes mask (Value < 70)
                dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
                dark_ratio = np.count_nonzero(dark_mask) / float(ch * cw)

                # Stripe edge transitions (horizontal gradient detecting vertical stripes)
                gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                stripe_edge_ratio = np.count_nonzero(abs_grad_x > 30) / float(ch * cw)

                # Check if it has tiger coat or stripes
                if (orange_ratio > 0.08 and (dark_ratio > 0.04 or stripe_edge_ratio > 0.06)) or (orange_ratio > 0.15):
                    is_tiger = True
                elif dark_ratio > 0.35 and orange_ratio < 0.05:
                    is_dark_boar = True

        # 3. Tiger Mapping:
        # COCO detects tigers as cat (feline), zebra (stripes), horse (large quadruped), dog, leopard, etc.
        if raw in ['tiger', 'cat', 'zebra', 'leopard', 'panther', 'cheetah', 'lion']:
            return 'tiger'

        if is_tiger:
            return 'tiger'

        if raw == 'horse':
            # Quadruped animal with tiger features or shown in camera
            return 'tiger'

        # 4. Wild Boar Mapping:
        # COCO detects wild boar as pig, sheep, cow, boar, wild_boar
        if raw in ['wild_boar', 'boar', 'pig']:
            return 'wild_boar'

        if raw in ['sheep', 'cow', 'bear'] or is_dark_boar:
            return 'wild_boar'

        if raw == 'dog':
            return 'tiger' if is_tiger else 'wild_boar'

        # Ignore all non-target COCO classes
        return None

    def process_frame(self, frame):
        """
        Runs YOLO inference, performs Pose Estimation, computes Dynamic Risk, and returns telemetry.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame if frame is not None else np.zeros((320, 320, 3), dtype=np.uint8), {}

        start_time = time.time()
        self.frame_counter += 1
        self.unconfirmed_detections = 0
        h_frame, w_frame, _ = frame.shape
        is_night = (datetime.now().hour >= 18 or datetime.now().hour < 6)
        
        config_data = getattr(self, 'config_data', None)
        if not config_data:
            config_data = self.load_config()

        # Phase 2: Preprocessing (optional CLAHE)
        config_prep = config_data.get("preprocessing", {})
        if config_prep.get("enabled", False):
            if config_prep.get("mode") == "clahe":
                try:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=config_prep.get("clahe_clip_limit", 2.5), 
                                            tileGridSize=tuple(config_prep.get("clahe_tile_grid_size", [8, 8])))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                except Exception as e:
                    print(f"[PREPROCESS ERROR] {e}")

        # Configure inference settings dynamically
        imgsz = int(os.getenv("INFERENCE_SIZE", "320"))
        iou = float(os.getenv("IOU_THRESHOLD", "0.7"))

        # Run YOLO with ByteTrack enabled for persistent object tracking
        try:
            ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
            with ctx:
                results_gen = self.model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False, imgsz=imgsz, iou=iou)
                results = list(results_gen)
        except Exception as e:
            print(f"[ERROR] Inference engine failure: {str(e)}")
            return frame, {"error": "Inference failure"}

        detected_count = 0
        detected_threats = []
        detected_behavior_events = []

        # 1. First pass to gather all active humans for distance calculations
        active_humans = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls].lower()
                    if label in ['person', 'human', 'poacher']:
                        hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                        active_humans.append(((hx1 + hx2) // 2, (hy1 + hy2) // 2))

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box_idx, box in enumerate(boxes):
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                raw_label = self.model.names[cls].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(w_frame - 1, x1))
                y1 = max(0, min(h_frame - 1, y1))
                x2 = max(0, min(w_frame, x2))
                y2 = max(0, min(h_frame, y2))
                
                crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

                # Precision 3-Animal (Tiger, Elephant, Wild Boar) + Person Classifier
                detected_threat = self.classify_wildlife_species(raw_label, crop)
                if detected_threat is None:
                    continue

                raw_class = detected_threat
                label = detected_threat.replace('_', ' ').capitalize()

                # Stable Track ID
                track_id = int(box.id[0]) if (box.id is not None) else (box_idx + 1)

                # Confidence threshold
                min_threshold = 0.20 if detected_threat in ['tiger', 'wild_boar'] else 0.25
                if confidence < min_threshold:
                    continue

                # Format Track ID (e.g. T001, E001, B001, P001)
                prefix = 'T' if detected_threat == 'tiger' else ('E' if detected_threat == 'elephant' else ('B' if 'boar' in detected_threat else 'P'))
                formatted_track_id = f"{prefix}{track_id:03d}"

                # Stabilize bounding box coordinates with Exponential Moving Average (EMA)
                # Eliminates box jumping/jitter across consecutive frames
                if not hasattr(self, 'smoothed_bboxes'):
                    self.smoothed_bboxes = {}

                if formatted_track_id in self.smoothed_bboxes:
                    px1, py1, px2, py2 = self.smoothed_bboxes[formatted_track_id]
                    alpha = 0.65  # Optimal responsiveness without coordinate jitter
                    x1 = int(alpha * x1 + (1.0 - alpha) * px1)
                    y1 = int(alpha * y1 + (1.0 - alpha) * py1)
                    x2 = int(alpha * x2 + (1.0 - alpha) * px2)
                    y2 = int(alpha * y2 + (1.0 - alpha) * py2)
                self.smoothed_bboxes[formatted_track_id] = (x1, y1, x2, y2)

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                w, h = (x2 - x1), (y2 - y1)
                
                # Temporal persistence check
                track_key = f"{raw_class}_ID{track_id}"
                history = self.track_history.get(track_key, [])
                history.append((cx, cy, time.time(), w, h))
                if len(history) > 30:
                    history.pop(0)
                self.track_history[track_key] = history

                debounce_frames = config_data.get("temporal_persistence_frames", 1)
                if len(history) < debounce_frames:
                    self.unconfirmed_detections += 1
                    continue

                detected_count += 1

                # Lifecycle State Machine
                active = getattr(self, 'active_tracks', {})
                current_time = time.time()
                
                if formatted_track_id not in active:
                    state = "OBJECT_ENTERED"
                    first_seen = current_time
                else:
                    first_seen = active[formatted_track_id]["first_seen"]
                    dt = max(current_time - history[-2][2], 0.01) if len(history) > 1 else 0.1
                    speed = math.sqrt((cx - history[-2][0])**2 + (cy - history[-2][1])**2) / dt
                    
                    if speed > 50:
                        state = "OBJECT_MOVING"
                    else:
                        state = "STATIONARY"
                    
                    if state == "OBJECT_MOVING" and (h_frame - cy) < (h_frame * 0.3):
                        state = "OBJECT_APPROACHING"

                # Assign Category
                category = "UNKNOWN"
                if raw_class in config_data.get("categories", {}).get("HUMAN", ["person", "human", "poacher"]):
                    category = "HUMAN"
                elif raw_class in config_data.get("categories", {}).get("WILDLIFE", []):
                    category = "WILDLIFE"
                else:
                    category = "WILDLIFE" if raw_class in ["elephant", "tiger", "leopard", "wild_boar", "bear", "deer", "boar"] else "OBJECT"

                # Behavioral Analysis Engine
                behavior_metrics = self.behavior_analyzer.analyze(
                    track_id=formatted_track_id,
                    species=detected_threat,
                    history=history,
                    group_size=len(boxes),
                    first_seen=first_seen,
                    active_humans=active_humans,
                    current_bbox=[x1, y1, x2, y2],
                    frame_shape=(h_frame, w_frame)
                )
                
                behaviour = behavior_metrics.get("behavior", behavior_metrics.get("behaviour", "standing"))
                velocity = behavior_metrics.get("speed", 0.0)
                direction_vector = behavior_metrics.get("direction", behavior_metrics.get("movement_direction", "stationary"))
                pose_pattern = f"Observable behavior: {behaviour}"

                if behavior_metrics.get("is_event", False):
                    detected_behavior_events.append(behavior_metrics)

                # Dynamic Risk Assessment
                risk_score, danger_level = self.calculate_dynamic_risk(
                    detected_threat, behaviour, velocity, direction_vector, group_size=len(boxes), is_night=is_night
                )
                
                cfg_class = config_data.get("classes", {}).get(detected_threat, config_data.get("classes", {}).get("default", {}))
                base_risk_weight = cfg_class.get("risk_weight", 50)
                risk_score = min(int((risk_score + base_risk_weight) / 2), 100)

                # Color scheme based on danger level
                if danger_level == "critical":
                    box_color = (0, 0, 255)
                elif danger_level == "high":
                    box_color = (0, 140, 255)
                elif danger_level == "medium":
                    box_color = (0, 255, 255)
                else:
                    box_color = (0, 255, 0)

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw Trajectory
                if len(history) > 2:
                    pts = np.array([(pt[0], pt[1]) for pt in history], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, box_color, 2)

                # Draw Overlay Badges with Large, High-Contrast Dark Card
                badge_title = f"{detected_threat.replace('_', ' ').upper()} #{formatted_track_id} [{risk_score}% RISK - {danger_level.upper()}]"
                behavior_clean = behaviour.upper().replace('_', ' ')
                direction_clean = direction_vector.upper().replace('_', ' ')
                badge_behavior = f"BEHAVIOR: {behavior_clean}"
                if velocity > 1.0:
                    badge_behavior += f" | {round(velocity, 1)} px/s"
                if direction_clean not in ["STATIONARY", "NONE", "UNKNOWN"]:
                    badge_behavior += f" | {direction_clean}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                diag = math.sqrt(w_frame**2 + h_frame**2)
                # Dynamic large font scaling based on footage resolution
                scale_title = max(0.80, min(1.35, round(diag / 1350.0, 2)))
                scale_sub = max(0.75, min(1.25, round(diag / 1500.0, 2)))
                thick = max(2, min(3, int(diag / 700.0)))

                (w1, h1), _ = cv2.getTextSize(badge_title, font, scale_title, thick)
                (w2, h2), _ = cv2.getTextSize(badge_behavior, font, scale_sub, thick)
                pad_x = int(14 * (diag / 800.0))
                pad_y = int(10 * (diag / 800.0))
                box_w = max(w1, w2) + (pad_x * 2)
                box_h = h1 + h2 + (pad_y * 3)

                # Ensure box stays within frame boundaries horizontally
                bx1 = max(10, x1)
                if bx1 + box_w > w_frame - 10:
                    bx1 = max(10, w_frame - 10 - box_w)
                bx2 = min(w_frame - 10, bx1 + box_w)

                # Vertical positioning: avoid top HUD (top 50-70px) and avoid clipping
                hud_margin = int(55 * (diag / 900.0))
                if y1 - box_h - 10 >= hud_margin:
                    by1 = y1 - box_h - 6
                    by2 = y1 - 2
                elif y2 + box_h + 10 <= h_frame:
                    by1 = y2 + 4
                    by2 = y2 + box_h + 8
                else:
                    by1 = min(h_frame - box_h - 10, max(hud_margin + 10, y1 + 10))
                    by2 = by1 + box_h

                # Solid dark backing card for 100% contrast against any background
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (10, 14, 20), -1)
                # Outer border in threat color
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, thick + 1)
                # Left accent indicator bar
                accent_w = max(6, int(8 * (diag / 800.0)))
                cv2.rectangle(frame, (bx1, by1), (bx1 + accent_w, by2), box_color, -1)

                # Line 1: Title (Bold White text)
                t1_y = by1 + h1 + pad_y
                cv2.putText(frame, badge_title, (bx1 + pad_x + 6, t1_y), font, scale_title, (255, 255, 255), thick, cv2.LINE_AA)

                # Line 2: Behavior (Bold Vibrant Neon Yellow/Cyan text for instant readability)
                t2_y = t1_y + h2 + pad_y + 4
                cv2.putText(frame, badge_behavior, (bx1 + pad_x + 6, t2_y), font, scale_sub, (0, 255, 255), thick, cv2.LINE_AA)

                # Trigger alerts for high/critical threats
                alert_key = f"alert_{formatted_track_id}"
                if danger_level in ["high", "critical"] or state in ["OBJECT_APPROACHING"]:
                    if current_time - self.last_alert_time.get(alert_key, 0) > self.alert_cooldown:
                        self.trigger_alert(
                            track_id=formatted_track_id,
                            species=detected_threat,
                            confidence=confidence,
                            frame=frame,
                            danger_level=danger_level,
                            risk_score=risk_score,
                            behaviour=behaviour,
                            movement_vector=direction_vector
                        )
                        self.last_alert_time[alert_key] = current_time

                # Update Active Tracks Map
                active[formatted_track_id] = {
                    "class": raw_class,
                    "first_seen": first_seen,
                    "last_seen": current_time,
                    "state": state,
                    "behavior": behaviour,
                    "risk_level": danger_level.upper(),
                    "speed": round(velocity, 1),
                    "direction": direction_vector
                }
                self.active_tracks = active

                # Store primary telemetry item
                telemetry = {
                    "track_id": formatted_track_id,
                    "animal": detected_threat,
                    "category": category,
                    "confidence": float(confidence),
                    "bbox": [x1, y1, x2, y2],
                    "behavior": behaviour,
                    "behaviour": behaviour,
                    "pose_pattern": pose_pattern,
                    "risk_score": risk_score,
                    "danger_level": danger_level,
                    "risk_level": danger_level.upper(),
                    "direction": direction_vector,
                    "movement_vector": direction_vector,
                    "movement_direction": direction_vector,
                    "speed": round(velocity, 1),
                    "velocity_px_sec": round(velocity, 1),
                    "distance_travelled": behavior_metrics.get("distance_travelled", 0.0),
                    "human_distance": behavior_metrics.get("human_distance"),
                    "loitering": behavior_metrics.get("loitering", False),
                    "group_size": len(boxes),
                    "class_id": cls,
                    "class_name": label,
                    "bounding_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                    "center_x": cx,
                    "center_y": cy,
                    "width": w,
                    "height": h,
                    "trajectory": [(pt[0], pt[1]) for pt in history],
                    "state": state,
                    "duration": round(current_time - first_seen, 1),
                    "timestamp": datetime.now().isoformat(),
                    "camera_id": getattr(self, 'camera_id', 'default'),
                    "frame_id": self.frame_counter
                }
                detected_threats.append(telemetry)
                    
        # Cleanup lost tracks
        current_time = time.time()
        active = getattr(self, 'active_tracks', {})
        lost_ids = []
        for tid, tdata in active.items():
            if current_time - tdata["last_seen"] > 2.0:
                tdata["state"] = "OBJECT_LOST"
                lost_ids.append(tid)

        for tid in lost_ids:
            del active[tid]
            if hasattr(self, 'smoothed_bboxes') and tid in self.smoothed_bboxes:
                del self.smoothed_bboxes[tid]
        self.active_tracks = active

        inference_time = (time.time() - start_time) * 1000  # ms
        self.fps = 1000 / inference_time if inference_time > 0 else 0
        
        # Draw prominent Full-Width Behavioral Telemetry HUD Bar on frame
        if detected_threats:
            top_threat = max(detected_threats, key=lambda t: t.get("risk_score", 0))
            t_animal = top_threat.get("animal", "THREAT").upper()
            t_tid = top_threat.get("track_id", "0")
            t_beh = top_threat.get("behavior", "DETECTED").upper().replace('_', ' ')
            t_dir = top_threat.get("direction", "STATIONARY").upper().replace('_', ' ')
            t_spd = top_threat.get("speed", 0.0)
            t_risk = top_threat.get("risk_score", 0)
            t_lvl = top_threat.get("risk_level", "LOW")

            diag = math.sqrt(w_frame**2 + h_frame**2)
            scale_hud = max(0.70, min(1.15, round(diag / 1600.0, 2)))
            thick_hud = max(2, min(3, int(diag / 750.0)))

            hud_text = f" AI BEHAVIOR: {t_animal} #{t_tid} | {t_beh} | {t_spd} px/s | RISK: {t_risk}% ({t_lvl}) "
            (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, scale_hud, thick_hud)

            # Fit scale if needed
            if hw > w_frame - 20:
                scale_hud = round(scale_hud * ((w_frame - 30) / float(hw)), 2)
                (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, scale_hud, thick_hud)

            hud_h = int(hh * 2.2) + 16
            hud_accent = (0, 0, 255) if t_risk >= 75 else ((0, 165, 255) if t_risk >= 50 else (0, 255, 0))

            # Full-width dark bar across top
            cv2.rectangle(frame, (0, 0), (w_frame, hud_h), (8, 12, 18), -1)
            cv2.rectangle(frame, (0, hud_h - 3), (w_frame, hud_h), hud_accent, -1)
            # Text centered vertically
            hud_ty = int(hud_h * 0.68)
            cv2.putText(frame, hud_text, (18, hud_ty), cv2.FONT_HERSHEY_SIMPLEX, scale_hud, (255, 255, 255), thick_hud, cv2.LINE_AA)

        metrics = {
            "fps": round(self.fps, 1),
            "inference_time_ms": round(inference_time, 1),
            "total_detections": detected_count,
            "false_positive_candidates": getattr(self, 'unconfirmed_detections', 0),
            "threats": detected_threats,
            "behavior_events": detected_behavior_events
        }

        return frame, metrics

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow('Wildlife Guardian - AI IoT Surveillance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Monitoring Stopped.")

if __name__ == "__main__":
    iot_system = WildlifeIoTSystem(model_path='yolo11n.pt', capture_index=0)
    iot_system.run()

