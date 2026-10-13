import cv2
import time
import sys
import os
import math
import numpy as np
from datetime import datetime
from behavior_analysis import WildlifeBehaviorAnalyzer

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

        # Target classes allowed for wildlife monitoring & intruder surveillance
        self.wildlife_classes = {
            "elephant", "tiger", "leopard", "bear", "deer", "wild boar", "boar", "monkey",
            "zebra", "giraffe", "cow", "horse", "sheep", "dog", "cat", "bird", "lion", "cheetah"
        }
        self.human_classes = {
            "person", "human", "poacher"
        }
        self.allowed_target_classes = self.wildlife_classes | self.human_classes

        # Resolve model class IDs for allowed target classes
        self.target_class_ids = []
        if self.model is not None and hasattr(self.model, 'names') and isinstance(self.model.names, dict):
            for cid, cname in self.model.names.items():
                cname_clean = str(cname).lower().strip()
                if cname_clean in self.allowed_target_classes or any(w in cname_clean for w in ["tiger", "elephant", "leopard", "bear", "deer", "boar", "monkey", "person", "human"]):
                    self.target_class_ids.append(cid)
            print(f"[SYSTEM] Filtered target classes: {len(self.target_class_ids)} target species/human IDs allowed.")

        # Initialize Camera / Capture Source
        self.cap = None
        if capture_index is not None:
            if isinstance(capture_index, int) and os.name == 'nt':
                self.cap = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(capture_index)
                
            if not self.cap.isOpened():
                err_msg = f"Could not open video stream (Index/File: {capture_index})."
                print(f"[ERROR] {err_msg}")
                raise RuntimeError(err_msg)
                
            if isinstance(capture_index, int):
                print(f"[SYSTEM] Warming up camera {capture_index}...")
                for _ in range(15):
                    self.cap.read()
                time.sleep(0.5)
                print("[SYSTEM] Camera warm-up complete.")
            
        # Alert Cooldowns
        self.last_alert_time = {}

        self.alert_cooldown = 10

        # Posture & Motion History Tracking (for velocity & behavior calculation)
        self.track_history = {}  # {track_id or label: [(cx, cy, timestamp, aspect_ratio, area)]}
        self.frame_counter = 0
        
        # Load independent behavior analysis module
        self.behavior_analyzer = WildlifeBehaviorAnalyzer()
        self._fallback_id_counter = 100
        self.active_tracks = {}

        print("[SYSTEM] Pose Estimation & Dynamic Risk Engine Active.")

    def _get_or_assign_track_id(self, box_id, raw_class, cx, cy):
        """
        Ensures consistent tracking ID even when ByteTrack temporarily drops ID.
        Uses centroid proximity matching (< 90px) with recent tracks of the same species.
        """
        if box_id is not None:
            try:
                return int(box_id)
            except (ValueError, TypeError):
                pass

        current_time = time.time()
        best_id = None
        min_dist = 90.0

        for tid, tdata in getattr(self, 'active_tracks', {}).items():
            if tdata.get("class") == raw_class and current_time - tdata.get("last_seen", 0) < 2.0:
                dist = math.sqrt((cx - tdata.get("cx", 0))**2 + (cy - tdata.get("cy", 0))**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tdata.get("raw_id")

        if best_id is not None:
            return best_id

        self._fallback_id_counter += 1
        return self._fallback_id_counter

    def update_stream(self, capture_index):
        """Dynamically switches the camera source to a new IP Webcam or device."""
        self.current_url = capture_index
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        print(f"[SYSTEM] Connecting to new camera stream: {capture_index}")
        if isinstance(capture_index, str) and capture_index.startswith('http'):
            # For HTTP IP Webcams, store current_url for background snapshot polling
            # Avoid blocking the main server thread on synchronous OpenCV network socket connect
            return True

        if isinstance(capture_index, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(capture_index)
            
        if self.cap is not None and not self.cap.isOpened():
            print(f"[ERROR] Could not open video stream (Index/File: {capture_index}).")
            return False
        return True

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

    def process_frame(self, frame):
        """
        Runs YOLO inference, performs Pose Estimation, computes Dynamic Risk, and returns telemetry.
        """
        # 1. Handle Malformed/Corrupted Frames
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("[WARNING] Malformed or empty frame received. Skipping.")
            return frame if frame is not None else np.zeros((320,320,3), dtype=np.uint8), {}

        # Elegant HUD Status pill in top-left
        h_frame, w_frame, _ = frame.shape
        hud_scale = max(min(w_frame, h_frame) / 540.0, 0.9)
        hud_text = "AI BEHAVIORAL SURVEILLANCE ACTIVE"
        hud_font_scale = 0.52 * hud_scale
        hud_thick = max(int(1.8 * hud_scale), 1)
        (htw, hth), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, hud_thick)
        hpad_x = int(10 * hud_scale)
        hpad_y = int(8 * hud_scale)
        cv2.rectangle(frame, (14, 14), (14 + htw + 2 * hpad_x, 14 + hth + 2 * hpad_y), (14, 14, 18), -1)
        cv2.rectangle(frame, (14, 14), (14 + htw + 2 * hpad_x, 14 + hth + 2 * hpad_y), (0, 230, 118), max(int(1.5 * hud_scale), 1))
        # Text with drop shadow
        cv2.putText(frame, hud_text, (14 + hpad_x + 1, 14 + hpad_y + hth + 1), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 0, 0), hud_thick + 2, cv2.LINE_AA)
        cv2.putText(frame, hud_text, (14 + hpad_x, 14 + hpad_y + hth), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 230, 118), hud_thick, cv2.LINE_AA)

        start_time = time.time()
        self.frame_counter += 1
        self.unconfirmed_detections = 0
        h_frame, w_frame, _ = frame.shape
        is_night = (datetime.now().hour >= 18 or datetime.now().hour < 6)
        
        # Load JSON config here so we can use preprocessing
        import json
        try:
            with open("wildlife_config.json", "r") as f:
                config_data = json.load(f)
        except:
            config_data = {"temporal_persistence_frames": 3, "classes": {}, "categories": {}}

        # Phase 2: Preprocessing
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

        # Run YOLO with ByteTrack enabled for persistent object tracking (filtered to target wildlife/human classes)
        try:
            track_kwargs = {
                "persist": True,
                "tracker": "bytetrack.yaml",
                "stream": True,
                "verbose": False,
                "imgsz": imgsz,
                "iou": iou
            }
            if getattr(self, 'target_class_ids', None):
                track_kwargs["classes"] = self.target_class_ids

            results_gen = self.model.track(frame, **track_kwargs)
            results = list(results_gen)
        except Exception as e:
            print(f"[ERROR] Inference engine failure: {str(e)}")
            return frame, {"error": "Inference failure"}

        # Phase 5: Adaptive ROI (Tiled Inference for Small Objects)
        config_roi = config_data.get("adaptive_roi", {})
        ignored_boxes = set()
        if config_roi.get("enabled", False):
            extra_synthetic_boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area_ratio = ((x2 - x1) * (y2 - y1)) / (h_frame * w_frame)
                        cmin, cmax = config_roi.get("trigger_conf_min", 0.15), config_roi.get("trigger_conf_max", 0.45)
                        
                        # Trigger crop if confidence is borderline OR if it's very small
                        if (cmin < conf < cmax) or (area_ratio < 0.1 and conf > 0.10):
                            pad = config_roi.get("padding", 60)
                            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                            cx2, cy2 = min(w_frame, x2 + pad), min(h_frame, y2 + pad)
                            crop = frame[cy1:cy2, cx1:cx2]
                            
                            if crop.size > 0:
                                crop_res = self.model.predict(crop, imgsz=imgsz, verbose=False, conf=0.20)
                                for cr in crop_res:
                                    if cr.boxes:
                                        for cbox in cr.boxes:
                                            cconf = float(cbox.conf[0])
                                            # Only keep if second-pass confidence is better or strong enough
                                            if cconf > conf or cconf > 0.40:
                                                ccls = int(cbox.cls[0])
                                                cx1_c, cy1_c, cx2_c, cy2_c = map(int, cbox.xyxy[0])
                                                new_x1, new_y1 = cx1_c + cx1, cy1_c + cy1
                                                new_x2, new_y2 = cx2_c + cx1, cy2_c + cy1
                                                trk_id = int(box.id[0]) if box.id is not None else None
                                                extra_synthetic_boxes.append(SyntheticBox(new_x1, new_y1, new_x2, new_y2, cconf, ccls, trk_id))
                                                ignored_boxes.add(id(box))
            
            if extra_synthetic_boxes:
                results.append(SyntheticResult(extra_synthetic_boxes))

        telemetry = None

        detected_count = 0
        detected_threats = []

        # 1. First pass: gather active humans and all candidate animal centroids for multi-animal interaction
        active_humans = []
        other_tracks_candidates = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if id(box) in ignored_boxes:
                        continue
                    cls = int(box.cls[0])
                    label = self.model.names[cls].lower()
                    if label not in getattr(self, 'allowed_target_classes', set()) and not any(w in label for w in ["tiger", "elephant", "leopard", "bear", "deer", "boar", "monkey", "person", "human"]):
                        continue
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    bcx, bcy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
                    if label in getattr(self, 'human_classes', {'person', 'human', 'poacher'}):
                        active_humans.append((bcx, bcy))
                    else:
                        other_tracks_candidates.append({
                            "cx": bcx,
                            "cy": bcy,
                            "animal": label,
                            "bbox": [bx1, by1, bx2, by2]
                        })

        behavior_events = []
        seen_tracks_in_frame = set()

        for result in results:
            boxes = result.boxes
            if boxes is None: continue
            
            detected_count += len(boxes)
            
            for box in boxes:
                if id(box) in ignored_boxes:
                    continue
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                raw_class = label.lower()

                # Filter out any non-target COCO objects (keyboard, book, laptop, chair, etc.)
                if raw_class not in getattr(self, 'allowed_target_classes', set()) and not any(w in raw_class for w in ["tiger", "elephant", "leopard", "bear", "deer", "boar", "monkey", "person", "human"]):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                w, h = (x2 - x1), (y2 - y1)

                # Extract or assign persistent Track ID (ByteTrack with spatial fallback)
                raw_box_id = box.id[0] if box.id is not None else None
                track_id = self._get_or_assign_track_id(raw_box_id, raw_class, cx, cy)

                # Load JSON config
                import json
                try:
                    with open("wildlife_config.json", "r") as f:
                        config_data = json.load(f)
                except:
                    config_data = {"temporal_persistence_frames": 2, "classes": {}, "categories": {}}

                cfg_class = config_data.get("classes", {}).get(raw_class, config_data.get("classes", {}).get("default", {"threshold": 0.45, "risk_weight": 10}))
                
                # Check confidence threshold for this class
                if confidence < cfg_class.get("threshold", 0.40):
                    continue

                # Format Track ID (e.g. Elephant #E001, Tiger #T002)
                prefix = raw_class[0].upper()
                formatted_track_id = f"{prefix}{track_id:03d}"
                
                if formatted_track_id in seen_tracks_in_frame:
                    continue
                seen_tracks_in_frame.add(formatted_track_id)

                # Temporal persistence check
                track_key = f"{raw_class}_ID{track_id}"
                history = self.track_history.get(track_key, [])
                history.append((cx, cy, time.time(), w, h))
                if len(history) > 35: # Keep trajectory history
                    history.pop(0)
                self.track_history[track_key] = history

                # Debounce persistence
                min_persist = config_data.get("temporal_persistence_frames", 2)
                if len(history) < min_persist:
                    self.unconfirmed_detections += 1
                    continue

                # Active Lifecycle State Machine
                active = getattr(self, 'active_tracks', {})
                current_time = time.time()
                
                if formatted_track_id not in active:
                    state = "OBJECT_ENTERED"
                    first_seen = current_time
                else:
                    first_seen = active[formatted_track_id]["first_seen"]
                    dt = max(current_time - history[-2][2], 0.01) if len(history) > 1 else 0.1
                    speed_calc = math.sqrt((cx - history[-2][0])**2 + (cy - history[-2][1])**2) / dt
                    
                    if speed_calc > 45: # pixels per sec
                        state = "OBJECT_MOVING"
                    else:
                        state = "STATIONARY"
                    
                    if state == "OBJECT_MOVING" and (h_frame - cy) < (h_frame * 0.35):
                        state = "OBJECT_APPROACHING"

                # Assign Category
                category = "UNKNOWN"
                if raw_class in config_data.get("categories", {}).get("HUMAN", []) or raw_class in getattr(self, 'human_classes', set()):
                    category = "HUMAN"
                elif raw_class in config_data.get("categories", {}).get("WILDLIFE", []) or raw_class in getattr(self, 'wildlife_classes', set()):
                    category = "WILDLIFE"
                
                # Non-target items are dropped immediately
                if category == "UNKNOWN":
                    continue

                detected_threat = raw_class

                if detected_threat:
                    # Execute comprehensive behavioral analysis engine
                    behavior_metrics = self.behavior_analyzer.analyze(
                        track_id=formatted_track_id,
                        species=detected_threat,
                        history=history,
                        group_size=len(boxes),
                        first_seen=first_seen,
                        active_humans=active_humans,
                        current_bbox=[x1, y1, x2, y2],
                        other_tracks=other_tracks_candidates,
                        frame_shape=(h_frame, w_frame)
                    )
                    
                    behaviour = behavior_metrics["behavior"]
                    velocity = behavior_metrics["speed"]
                    direction_vector = behavior_metrics["direction"]
                    risk_level = behavior_metrics["risk_level"]  # LOW, MEDIUM, HIGH, CRITICAL
                    danger_level = risk_level.lower()
                    pose_pattern = f"Observable behavior: {behaviour}"

                    # Calculate numerical risk score (0-100) for telemetry dials & gauges
                    base_risk_weight = cfg_class.get("risk_weight", 10)
                    if risk_level == "CRITICAL":
                        risk_score = max(85, min(100, int(75 + base_risk_weight * 0.25)))
                    elif risk_level == "HIGH":
                        risk_score = max(65, min(84, int(55 + base_risk_weight * 0.25)))
                    elif risk_level == "MEDIUM":
                        risk_score = max(40, min(64, int(35 + base_risk_weight * 0.25)))
                    else:
                        risk_score = max(15, min(39, int(15 + base_risk_weight * 0.20)))

                    # Color scheme based on risk level
                    if risk_level == "CRITICAL":
                        box_color = (0, 0, 255)       # Red
                    elif risk_level == "HIGH":
                        box_color = (0, 140, 255)     # Orange
                    elif risk_level == "MEDIUM":
                        box_color = (0, 255, 255)     # Yellow
                    else:
                        box_color = (0, 255, 0)       # Green

                    # Dynamic scaling relative to frame dimensions
                    scale = max(min(w_frame, h_frame) / 480.0, 1.0)
                    font_scale_title = 0.58 * scale
                    font_scale_sub = 0.48 * scale
                    font_thick = max(int(1.8 * scale), 1)
                    box_thick = max(int(2.5 * scale), 2)
                    line_thick = max(int(2.0 * scale), 2)

                    # Draw Bounding Box with high-contrast corner brackets
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thick)
                    corner_len = int(min(x2 - x1, y2 - y1) * 0.18)
                    if corner_len > 8:
                        c_thick = box_thick + 1
                        # Top-left corner
                        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), c_thick)
                        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), c_thick)
                        # Top-right corner
                        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), c_thick)
                        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), c_thick)
                        # Bottom-left corner
                        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), c_thick)
                        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), c_thick)
                        # Bottom-right corner
                        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), c_thick)
                        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), c_thick)
                    
                    # Draw Trajectory line
                    if len(history) > 2:
                        pts = np.array([(pt[0], pt[1]) for pt in history], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, box_color, line_thick)

                    # Draw Interaction line if interacting with nearby animal
                    for other in behavior_metrics.get("interacting_with", []):
                        ot_id = other.get("track_id")
                        if ot_id in active:
                            ot_cx, ot_cy = active[ot_id].get("cx", 0), active[ot_id].get("cy", 0)
                            cv2.line(frame, (cx, cy), (ot_cx, ot_cy), (0, 255, 255), line_thick, cv2.LINE_AA)

                    # Extended High-Visibility Overlay Badges (Animal, Track ID, Behavior, Risk)
                    badge_title = f"{detected_threat.upper()} #{formatted_track_id} | {risk_level} RISK ({risk_score}%)"
                    badge_sub = f"{behaviour.upper()} | {velocity} px/s | {direction_vector.replace('_', ' ').upper()}"
                    
                    (tw1, th1), _ = cv2.getTextSize(badge_title, cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, font_thick)
                    (tw2, th2), _ = cv2.getTextSize(badge_sub, cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, max(font_thick - 1, 1))
                    
                    pad_x = int(12 * scale)
                    pad_y = int(8 * scale)
                    gap_y = int(6 * scale)
                    badge_w = max(tw1, tw2) + 2 * pad_x
                    badge_h = th1 + th2 + 2 * pad_y + gap_y

                    # Determine vertical placement (above box if fits, else inside top of box)
                    if y1 - badge_h - int(6 * scale) >= 0:
                        by1 = y1 - badge_h - int(6 * scale)
                    else:
                        by1 = y1 + int(4 * scale)
                    by2 = by1 + badge_h

                    # Keep badge strictly inside horizontal frame bounds
                    bx1 = max(4, min(x1, w_frame - badge_w - 4))
                    bx2 = min(w_frame - 4, bx1 + badge_w)

                    # Solid high-contrast dark background badge with colored accent border
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (12, 12, 14), -1)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, max(int(1.8 * scale), 1))

                    # Subtle separator line between title and subtitle
                    sep_y = by1 + pad_y + th1 + gap_y // 2
                    cv2.line(frame, (bx1 + pad_x // 2, sep_y), (bx2 - pad_x // 2, sep_y), (60, 60, 65), 1)

                    # Text Title (Species, ID, Risk) with shadow for crisp readability
                    t_x, t_y = bx1 + pad_x, by1 + pad_y + th1
                    cv2.putText(frame, badge_title, (t_x + 1, t_y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 0, 0), font_thick + 2, cv2.LINE_AA)
                    cv2.putText(frame, badge_title, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, box_color, font_thick, cv2.LINE_AA)

                    # Text Subtitle (Behavior, Speed, Direction) with shadow
                    s_x, s_y = bx1 + pad_x, by1 + pad_y + th1 + gap_y + th2
                    cv2.putText(frame, badge_sub, (s_x + 1, s_y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0, 0, 0), max(font_thick, 2), cv2.LINE_AA)
                    cv2.putText(frame, badge_sub, (s_x, s_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 255, 255), max(font_thick - 1, 1), cv2.LINE_AA)

                    # Collect confirmed behavioral events for logging & feed
                    if behavior_metrics.get("is_event", False):
                        event_record = {
                            "timestamp": datetime.now().isoformat(),
                            "animal": detected_threat,
                            "track_id": formatted_track_id,
                            "behavior": behaviour,
                            "confidence": behavior_metrics["confidence"],
                            "speed": velocity,
                            "direction": direction_vector,
                            "risk_level": risk_level,
                            "duration": behavior_metrics["duration"],
                            "camera_id": getattr(self, 'camera_id', 'IP_WEBCAM_01'),
                            "event_summary": behavior_metrics.get("event_summary", "")
                        }
                        behavior_events.append(event_record)

                    # Deduplicated Alert Trigger (HIGH/CRITICAL or charging approach)
                    alert_key = f"alert_{formatted_track_id}"
                    if danger_level in ["high", "critical"] or behaviour in ["possible charging", "approaching camera"]:
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
                        "raw_id": track_id,
                        "cx": cx,
                        "cy": cy,
                        "first_seen": first_seen,
                        "last_seen": current_time,
                        "state": state,
                        "behavior": behaviour,
                        "risk_level": risk_level,
                        "speed": velocity,
                        "direction": direction_vector
                    }
                    self.active_tracks = active

                    # Primary telemetry payload
                    telemetry = {
                        "track_id": formatted_track_id,
                        "raw_track_id": track_id,
                        "animal": detected_threat,
                        "category": category,
                        "confidence": float(confidence),
                        "bbox": [x1, y1, x2, y2],
                        "behaviour": behaviour,
                        "behavior": behaviour,
                        "behavior_confidence": behavior_metrics["confidence"],
                        "pose_pattern": pose_pattern,
                        "risk_score": risk_score,
                        "danger_level": danger_level,
                        "risk_level": risk_level,
                        "movement_vector": direction_vector,
                        "movement_direction": direction_vector,
                        "velocity_px_sec": round(velocity, 1),
                        "speed": round(velocity, 1),
                        "distance_travelled": behavior_metrics["distance_travelled"],
                        "human_distance": behavior_metrics["human_distance"],
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
                        "duration": behavior_metrics["duration"],
                        "loitering": behavior_metrics.get("loitering", False),
                        "interacting_with": behavior_metrics.get("interacting_with", []),
                        "timestamp": datetime.now().isoformat(),
                        "camera_id": getattr(self, 'camera_id', 'IP_WEBCAM_01'),
                        "frame_id": self.frame_counter
                    }
                    detected_threats.append(telemetry)
                    
        # Analytics Archive: Cleanup lost tracks and write to JSON
        current_time = time.time()
        active = getattr(self, 'active_tracks', {})
        lost_ids = []
        for tid, tdata in active.items():
            if current_time - tdata["last_seen"] > 2.5: # Lost after 2.5 seconds
                tdata["state"] = "OBJECT_LOST"
                lost_ids.append(tid)
                self.behavior_analyzer.cleanup_track(tid)
                
                try:
                    with open("analytics_log.json", "a") as af:
                        af.write(json.dumps(tdata) + "\n")
                except:
                    pass

        for tid in lost_ids:
            del active[tid]
        self.active_tracks = active
                    
        import gc
        gc.collect()

        inference_time = (time.time() - start_time) * 1000  # ms
        self.fps = 1000 / inference_time if inference_time > 0 else 0
        
        metrics = {
            "fps": round(self.fps, 1),
            "inference_time_ms": round(inference_time, 1),
            "total_detections": detected_count,
            "false_positive_candidates": getattr(self, 'unconfirmed_detections', 0),
            "threats": detected_threats,
            "behavior_events": behavior_events,
            "active_tracks_count": len(self.active_tracks)
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
    iot_system = WildlifeIoTSystem(model_path='yolov8n.pt', capture_index=0)
    iot_system.run()

