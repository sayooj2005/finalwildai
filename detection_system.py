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

        print("[SYSTEM] Pose Estimation & Dynamic Risk Engine Active.")

    def update_stream(self, capture_index):
        """Dynamically switches the camera source to a new IP Webcam or device."""
        self.current_url = capture_index
        if self.cap is not None:
            self.cap.release()
        
        print(f"[SYSTEM] Connecting to new camera stream: {capture_index}")
        if isinstance(capture_index, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(capture_index)
            
        if not self.cap.isOpened():
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

        cv2.putText(frame, "YOLO PIPELINE ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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

        # Run YOLO with ByteTrack enabled for persistent object tracking
        try:
            results_gen = self.model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False, imgsz=imgsz, iou=iou)
            results = list(results_gen)
        except Exception as e:
            print(f"[ERROR] Inference engine failure: {str(e)}")
            return frame, {"error": "Inference failure"}

        # Phase 5: Adaptive ROI (Tiled Inference for Small Objects)
        config_roi = config_data.get("adaptive_roi", {})
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
                                                
                                                # Overwrite original box confidence so we don't process it as low-conf
                                                box.conf[0] = 0.0 
            
            if extra_synthetic_boxes:
                results.append(SyntheticResult(extra_synthetic_boxes))

        telemetry = None

        detected_count = 0
        detected_threats = []

        # 1. First pass to gather all active humans for distance calculations
        active_humans = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls].lower()
                    if label in ['person', 'human', 'poacher']:
                        hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                        active_humans.append( (int((hx1+hx2)/2), int((hy1+hy2)/2)) )
                        
        for result in results:
            boxes = result.boxes
            if boxes is None: continue
            
            detected_count += len(boxes)
            
            for box in boxes:
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                # Extract ByteTrack ID (if tracking is successful, otherwise fallback)
                track_id = int(box.id[0]) if box.id is not None else int(time.time() * 1000) % 10000

                # Load JSON config
                import json
                try:
                    with open("wildlife_config.json", "r") as f:
                        config_data = json.load(f)
                except:
                    config_data = {"temporal_persistence_frames": 3, "classes": {}, "categories": {}}

                raw_class = label.lower()
                cfg_class = config_data.get("classes", {}).get(raw_class, config_data.get("classes", {}).get("default", {"threshold": 0.5, "risk_weight": 10}))
                
                # Check confidence threshold for THIS class
                if confidence < cfg_class.get("threshold", 0.5):
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                w, h = (x2 - x1), (y2 - y1)

                # Format Track ID (e.g. Tiger #T001)
                prefix = raw_class[0].upper()
                formatted_track_id = f"{prefix}{track_id:03d}"
                
                # Temporal persistence check
                track_key = f"{raw_class}_ID{track_id}"
                history = self.track_history.get(track_key, [])
                history.append((cx, cy, time.time(), w, h))
                if len(history) > 30: # Keep longer history for drawing trajectories
                    history.pop(0)
                self.track_history[track_key] = history

                # Only confirm if it appeared enough times (debounce)
                if len(history) < config_data.get("temporal_persistence_frames", 3):
                    self.unconfirmed_detections += 1
                    continue

                # Advanced Lifecycle State Machine
                active = getattr(self, 'active_tracks', {})
                current_time = time.time()
                
                if formatted_track_id not in active:
                    state = "OBJECT_ENTERED"
                    first_seen = current_time
                else:
                    first_seen = active[formatted_track_id]["first_seen"]
                    prev_state = active[formatted_track_id]["state"]
                    # Calculate speed
                    dt = max(current_time - history[-2][2], 0.01) if len(history)>1 else 0.1
                    speed = math.sqrt((cx-history[-2][0])**2 + (cy-history[-2][1])**2) / dt
                    
                    if speed > 50: # pixels per sec
                        state = "OBJECT_MOVING"
                    else:
                        state = "STATIONARY"
                    
                    # Approaching logic
                    if state == "OBJECT_MOVING" and (h_frame - cy) < (h_frame * 0.3): # getting close to bottom of frame (camera)
                        state = "OBJECT_APPROACHING"

                # Assign Category
                category = "UNKNOWN"
                if raw_class in config_data.get("categories", {}).get("HUMAN", []):
                    category = "HUMAN"
                elif raw_class in config_data.get("categories", {}).get("WILDLIFE", []):
                    category = "WILDLIFE"
                
                detected_threat = raw_class

                if detected_threat:
                    # Execute independent behavior analysis
                    behavior_metrics = self.behavior_analyzer.analyze(
                        track_id=formatted_track_id,
                        species=detected_threat,
                        history=history,
                        group_size=len(boxes),
                        first_seen=first_seen,
                        active_humans=active_humans
                    )
                    
                    behaviour = behavior_metrics["behaviour"]
                    velocity = behavior_metrics["speed"]
                    direction_vector = behavior_metrics["movement_direction"]
                    pose_pattern = f"Observable behavior: {behaviour}"

                    # Dynamic Risk Assessment
                    risk_score, danger_level = self.calculate_dynamic_risk(
                        detected_threat, behaviour, velocity, direction_vector, group_size=len(boxes), is_night=is_night
                    )
                    
                    base_risk_weight = cfg_class.get("risk_weight", 10)
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
                        pts = np.array([(pt[0], pt[1]) for pt in history], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, box_color, 2)

                    # Draw Overlay Badges (Using Formatted Track ID)
                    badge_text = f"{detected_threat.upper()} #{formatted_track_id} [{risk_score}% RISK]"
                    cv2.rectangle(frame, (x1, max(y1 - 40, 0)), (x1 + len(badge_text)*10 + 10, y1), box_color, -1)
                    cv2.putText(frame, badge_text, (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    sub_badge = f"{state} | {behaviour}"
                    cv2.putText(frame, sub_badge, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Deduplicated Alert Logic (Trigger only if State changed to high threat or cooldown expired)
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
                        "state": state
                    }
                    self.active_tracks = active

                    # Store primary telemetry item
                    telemetry = {
                        "track_id": formatted_track_id, # Replaces raw int track_id
                        "animal": detected_threat,
                        "category": category,
                        "confidence": float(confidence),
                        "bbox": [x1, y1, x2, y2],
                        "behaviour": behaviour,
                        "pose_pattern": pose_pattern,
                        "risk_score": risk_score,
                        "danger_level": danger_level,
                        "movement_vector": direction_vector,
                        "velocity_px_sec": round(velocity, 1),
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
                        "duration": round(current_time - first_seen, 1),
                        "timestamp": datetime.now().isoformat(),
                        "camera_id": getattr(self, 'camera_id', 'default'),
                        "frame_id": self.frame_counter
                    }
                    detected_threats.append(telemetry)
                    
        # Analytics Archive: Cleanup lost tracks and write to JSON
        current_time = time.time()
        active = getattr(self, 'active_tracks', {})
        lost_ids = []
        for tid, tdata in active.items():
            if current_time - tdata["last_seen"] > 2.0: # Lost after 2 seconds
                tdata["state"] = "OBJECT_LOST"
                lost_ids.append(tid)
                
                # Append to analytics log
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
            "threats": detected_threats
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

