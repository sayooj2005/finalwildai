import math
import time
from datetime import datetime

# ==============================================================================
# CONFIGURABLE BEHAVIORAL THRESHOLDS
# Computer-vision heuristics for inferring movement patterns and risk indicators.
# Not scientifically validated biological/emotional models.
# ==============================================================================
BEHAVIOR_CONFIG = {
    # Speed thresholds (pixels per second)
    "stationary_threshold": 20.0,       # Below this: stationary / resting / standing
    "walking_threshold": 75.0,          # Between stationary and walking: walking
    "running_threshold": 150.0,         # Above walking up to this or more: running
    "charging_speed_threshold": 160.0,  # Fast speed approaching camera: possible charging

    # Bounding-box looming / area expansion rate (% increase per second)
    "approach_area_growth_rate": 0.12,  # BBox area increasing >12%/sec indicates approach towards camera
    "retreat_area_shrink_rate": -0.10,  # BBox area decreasing <-10%/sec indicates moving away from camera
    "rapid_approach_area_rate": 0.25,   # Rapid area growth >25%/sec indicates rapid approach

    # Centroid vertical movement thresholds (pixels/sec in image space, +y is downward toward camera)
    "approach_dy_threshold": 25.0,      # Significant downward movement toward bottom of frame
    "retreat_dy_threshold": -25.0,      # Significant upward movement toward top of frame

    # Temporal & Spatial thresholds
    "min_track_duration": 0.8,          # Seconds before confirming dynamic movement
    "loitering_duration": 5.0,          # Seconds in scene within loitering radius to flag loitering
    "loitering_radius": 60.0,           # Pixel radius around origin to count as loitering
    "resting_duration": 6.0,            # Seconds stationary before transitioning from standing to resting
    "sudden_accel_threshold": 140.0,    # Acceleration spike (px/s^2) for sudden movement
    "repetition_window_sec": 4.0,       # Window to check for repeated direction reversal
    "repetition_reversals": 3,          # Number of direction flips to flag repeated/pacing movement

    # Multi-animal interaction
    "interaction_proximity_px": 140.0,  # Proximity between animals to detect interaction
    
    # Temporal smoothing and stability
    "smoothing_window": 5,              # Number of history points for moving averages
    "behavior_confirmation_frames": 3,  # Consecutive frames a candidate behavior must persist before confirmation
    "risk_cooldown": 8.0,               # Cooldown seconds between repeated high/critical alert events
}


class WildlifeBehaviorAnalyzer:
    """
    Independent behavioral-analysis module for wildlife surveillance.
    Infers movement-based behaviors from object detection bounding boxes, tracking centroids,
    and temporal trajectory history over consecutive frames.

    NOTE: Classifications are computer-vision behavioral indicators based on trajectory,
    speed, bounding box looming, and spatial proximity. They do NOT claim biological
    or emotional understanding.
    """

    def __init__(self, config=None):
        self.config = dict(BEHAVIOR_CONFIG)
        if config:
            self.config.update(config)

        # Per-track internal state for temporal stability and hysteresis:
        # {track_id: {
        #     "confirmed_behavior": str,
        #     "candidate_behavior": str,
        #     "candidate_count": int,
        #     "confirmed_risk": str,
        #     "origin_pos": (cx, cy),
        #     "last_event_time": float,
        #     "last_event_behavior": str,
        #     "stationary_start_time": float or None,
        #     "recent_directions": [(dx, dy, timestamp)],
        #     "prev_speed": float
        # }}
        self.track_states = {}

    def update_config(self, new_config: dict):
        """Updates behavioral configuration thresholds dynamically."""
        self.config.update(new_config)

    def get_config(self) -> dict:
        """Returns the current behavioral configuration dictionary."""
        return dict(self.config)

    def _calculate_distance(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def _calculate_smoothed_metrics(self, history):
        """
        Computes smoothed velocity, acceleration, and bounding box area growth rate
        using a moving window across recent history points.
        Each history item is (cx, cy, timestamp, w, h).
        """
        if len(history) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        window_size = min(len(history), self.config["smoothing_window"])
        recent = history[-window_size:]

        # Overall displacement across the window
        dt_total = max(recent[-1][2] - recent[0][2], 0.05)
        dx_total = recent[-1][0] - recent[0][0]
        dy_total = recent[-1][1] - recent[0][1]

        vx = dx_total / dt_total
        vy = dy_total / dt_total
        speed = math.sqrt(vx**2 + vy**2)

        # Instantaneous delta between last two frames for acceleration
        dt_last = max(history[-1][2] - history[-2][2], 0.02)
        inst_speed = math.sqrt(
            (history[-1][0] - history[-2][0]) ** 2 + (history[-1][1] - history[-2][1]) ** 2
        ) / dt_last

        # Area growth rate: (area_now - area_start) / (area_start * dt)
        area_start = max(recent[0][3] * recent[0][4], 1.0)
        area_now = max(recent[-1][3] * recent[-1][4], 1.0)
        area_growth_rate = (area_now - area_start) / (area_start * dt_total)

        # Total cumulative distance traveled across all history
        cum_distance = 0.0
        for i in range(1, len(history)):
            cum_distance += self._calculate_distance(
                (history[i - 1][0], history[i - 1][1]),
                (history[i][0], history[i][1]),
            )

        return speed, inst_speed, vx, vy, area_growth_rate, cum_distance

    def _determine_direction(self, vx, vy, speed, area_growth_rate):
        """Categorizes movement direction relative to camera and frame axes."""
        if speed < self.config["stationary_threshold"]:
            return "stationary"

        # Check looming effect (bounding box expanding or contracting significantly)
        if area_growth_rate > self.config["approach_area_growth_rate"] and vy > 0:
            return "toward_camera"
        elif area_growth_rate < self.config["retreat_area_shrink_rate"] and vy < 0:
            return "away_from_camera"

        # Direction based on displacement vector
        if abs(vy) > abs(vx) * 1.2:
            return "toward_camera" if vy > 0 else "away_from_camera"
        elif abs(vx) > abs(vy) * 1.2:
            return "moving_right" if vx > 0 else "moving_left"
        else:
            vertical = "toward_camera" if vy > 0 else "away_from_camera"
            horizontal = "right" if vx > 0 else "left"
            return f"{vertical}_{horizontal}"

    def _check_repeated_movement(self, state, vx, vy, current_time):
        """Detects pacing or repeated back-and-forth movement across a time window."""
        recent_dirs = state.get("recent_directions", [])
        # Record primary axis direction sign: +1, -1, or 0
        primary_dir = 1 if vx > 15 else (-1 if vx < -15 else 0)
        recent_dirs.append((primary_dir, current_time))

        # Prune old directions outside window
        window_sec = self.config["repetition_window_sec"]
        recent_dirs = [d for d in recent_dirs if current_time - d[1] <= window_sec]
        state["recent_directions"] = recent_dirs

        if len(recent_dirs) < 4:
            return False

        # Count sign flips
        reversals = 0
        prev_sign = None
        for sign, _ in recent_dirs:
            if sign != 0:
                if prev_sign is not None and sign != prev_sign:
                    reversals += 1
                prev_sign = sign

        return reversals >= self.config["repetition_reversals"]

    def _evaluate_risk_level(self, species, behavior, speed, direction, group_size):
        """
        Calculates CV behavioral risk level: LOW, MEDIUM, HIGH, CRITICAL.
        Combines species threat profile with observable behavior and velocity.
        """
        species_lower = species.lower()
        is_predator = any(p in species_lower for p in ["tiger", "leopard", "panther"])
        is_large_threat = any(t in species_lower for t in ["elephant", "bear", "poacher", "human"])

        # Default baseline
        risk_level = "LOW"

        if behavior == "possible charging":
            risk_level = "CRITICAL"
        elif behavior == "approaching camera":
            if is_predator or is_large_threat:
                risk_level = "HIGH" if speed < self.config["running_threshold"] else "CRITICAL"
            elif speed > self.config["walking_threshold"]:
                risk_level = "HIGH"
            else:
                risk_level = "MEDIUM"
        elif behavior in ["running", "sudden movement"]:
            if is_predator or is_large_threat:
                risk_level = "HIGH"
            else:
                risk_level = "MEDIUM"
        elif behavior in ["loitering / prolonged presence", "repeated movement", "multiple-animal interaction"]:
            if is_predator:
                risk_level = "HIGH"
            elif is_large_threat and group_size >= 2:
                risk_level = "HIGH"
            else:
                risk_level = "MEDIUM"
        elif behavior in ["walking", "normal movement"]:
            if is_predator and direction == "toward_camera":
                risk_level = "HIGH"
            elif is_predator:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
        elif behavior in ["resting / idle", "standing", "moving away from camera"]:
            if is_predator:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

        return risk_level

    def analyze(
        self,
        track_id,
        species,
        history,
        group_size=1,
        first_seen=None,
        active_humans=None,
        current_bbox=None,
        other_tracks=None,
        frame_shape=(720, 1280)
    ):
        """
        Main analysis method per tracked animal.
        Accepts:
            - track_id: Unique tracking ID
            - species: Detected animal class label
            - history: Trajectory list [(cx, cy, timestamp, w, h), ...]
            - group_size: Total count of animals detected in current frame
            - first_seen: Timestamp when track first appeared
            - active_humans: Optional list of (hx, hy) coordinates of humans detected
            - current_bbox: Current bounding box [x1, y1, x2, y2]
            - other_tracks: List of dicts for other active tracks in the frame
            - frame_shape: (height, width) of the video frame
        Returns:
            Dictionary matching standard behavioral analysis schema.
        """
        current_time = time.time()
        first_seen = first_seen if first_seen is not None else current_time
        duration = max(round(current_time - first_seen, 1), 0.1)

        # Initialize or retrieve persistent track state for temporal stability
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "confirmed_behavior": "standing",
                "candidate_behavior": "standing",
                "candidate_count": 0,
                "confirmed_risk": "LOW",
                "origin_pos": (history[-1][0], history[-1][1]) if history else (0, 0),
                "last_event_time": 0.0,
                "last_event_behavior": "",
                "stationary_start_time": current_time,
                "recent_directions": [],
                "prev_speed": 0.0,
            }
        state = self.track_states[track_id]

        # Handle early tracking warmup (< 2 points)
        if len(history) < 2:
            return {
                "track_id": track_id,
                "animal": species,
                "behavior": state["confirmed_behavior"],
                "confidence": 0.5,
                "speed": 0.0,
                "direction": "stationary",
                "duration": duration,
                "risk_level": "LOW",
                "distance_travelled": 0.0,
                "human_distance": None,
                "timestamp": datetime.now().isoformat(),
                "is_event": False,
                "event_summary": f"[{datetime.now().strftime('%H:%M:%S')}] {species.capitalize()} #{track_id} -> {state['confirmed_behavior']} -> LOW",
                "interacting_with": [],
            }

        cx, cy, t, w, h = history[-1]
        speed, inst_speed, vx, vy, area_growth_rate, cum_dist = self._calculate_smoothed_metrics(history)

        # Acceleration check (sudden movement)
        prev_speed = state.get("prev_speed", speed)
        dt_last = max(history[-1][2] - history[-2][2], 0.02)
        acceleration = abs(inst_speed - prev_speed) / dt_last
        state["prev_speed"] = inst_speed

        # Movement direction
        direction = self._determine_direction(vx, vy, speed, area_growth_rate)

        # Human Proximity calculation
        min_human_dist = None
        if active_humans:
            for hx, hy in active_humans:
                d = self._calculate_distance((cx, cy), (hx, hy))
                if min_human_dist is None or d < min_human_dist:
                    min_human_dist = round(d, 1)

        # Multi-animal interaction check
        interacting_tracks = []
        if other_tracks:
            for ot in other_tracks:
                if ot.get("track_id") != track_id:
                    ot_cx, ot_cy = ot.get("cx", 0), ot.get("cy", 0)
                    dist_to_other = self._calculate_distance((cx, cy), (ot_cx, ot_cy))
                    if dist_to_other < self.config["interaction_proximity_px"]:
                        interacting_tracks.append({
                            "track_id": ot.get("track_id"),
                            "species": ot.get("animal", "animal"),
                            "distance": round(dist_to_other, 1)
                        })

        # Loitering calculation (prolonged stay in small area)
        origin_cx, origin_cy = state["origin_pos"]
        dist_from_origin = self._calculate_distance((cx, cy), (origin_cx, origin_cy))
        is_loitering = (duration >= self.config["loitering_duration"]) and (dist_from_origin <= self.config["loitering_radius"])

        # Pacing / Repeated movement check
        is_repeated = self._check_repeated_movement(state, vx, vy, current_time)

        # ======================================================================
        # BEHAVIOR CLASSIFICATION LOGIC
        # Practical CV inference based on velocity, bounding-box expansion, and spatial patterns
        # ======================================================================
        raw_behavior = "normal movement"
        confidence = 0.70

        # Check for possible charging (rapid approach + high speed + booming bbox or downward vector)
        is_approaching_camera = (
            direction == "toward_camera"
            or area_growth_rate >= self.config["approach_area_growth_rate"]
            or vy >= self.config["approach_dy_threshold"]
        )

        is_moving_away = (
            direction == "away_from_camera"
            or area_growth_rate <= self.config["retreat_area_shrink_rate"]
            or vy <= self.config["retreat_dy_threshold"]
        )

        if speed >= self.config["charging_speed_threshold"] and (is_approaching_camera or area_growth_rate >= self.config["rapid_approach_area_rate"]):
            raw_behavior = "possible charging"
            confidence = 0.90
        elif acceleration >= self.config["sudden_accel_threshold"] and duration >= self.config["min_track_duration"]:
            raw_behavior = "sudden movement"
            confidence = 0.85
        elif speed >= self.config["running_threshold"]:
            raw_behavior = "running"
            confidence = 0.88
        elif speed < self.config["stationary_threshold"]:
            # Stationary: distinction between standing and resting/idle
            if state["stationary_start_time"] is None:
                state["stationary_start_time"] = current_time

            stationary_duration = current_time - state["stationary_start_time"]
            if stationary_duration >= self.config["resting_duration"]:
                raw_behavior = "resting / idle"
                confidence = 0.85
            else:
                raw_behavior = "standing"
                confidence = 0.80
        else:
            # Animal is moving with moderate speed
            state["stationary_start_time"] = None  # reset stationary timer

            if is_approaching_camera:
                raw_behavior = "approaching camera"
                confidence = 0.86
            elif is_moving_away:
                raw_behavior = "moving away from camera"
                confidence = 0.84
            elif is_repeated:
                raw_behavior = "repeated movement"
                confidence = 0.82
            elif is_loitering:
                raw_behavior = "loitering / prolonged presence"
                confidence = 0.80
            elif interacting_tracks:
                raw_behavior = "multiple-animal interaction"
                confidence = 0.82
            elif speed >= self.config["stationary_threshold"] and speed < self.config["walking_threshold"]:
                raw_behavior = "walking"
                confidence = 0.78
            else:
                raw_behavior = "normal movement"
                confidence = 0.75

        # ======================================================================
        # TEMPORAL STABILITY & HYSTERESIS (CONSECUTIVE-FRAME CONFIRMATION)
        # Prevents flickering/jitter between frames
        # ======================================================================
        prev_confirmed = state["confirmed_behavior"]

        if raw_behavior == prev_confirmed:
            state["candidate_behavior"] = raw_behavior
            state["candidate_count"] = 0
        else:
            if raw_behavior == state["candidate_behavior"]:
                state["candidate_count"] += 1
            else:
                state["candidate_behavior"] = raw_behavior
                state["candidate_count"] = 1

            # High-threat sudden behaviors (charging, sudden movement) require only 2 frames, others use configured frames
            required_confirmations = 2 if raw_behavior in ["possible charging", "sudden movement"] else self.config["behavior_confirmation_frames"]
            if state["candidate_count"] >= required_confirmations:
                state["confirmed_behavior"] = raw_behavior
                state["candidate_count"] = 0

        confirmed_behavior = state["confirmed_behavior"]

        # Evaluate risk level based on confirmed behavior
        risk_level = self._evaluate_risk_level(species, confirmed_behavior, speed, direction, group_size)
        state["confirmed_risk"] = risk_level

        # Check if this qualifies as a confirmed behavioral event (transition or high-risk alert)
        is_event = False
        time_since_last_event = current_time - state.get("last_event_time", 0.0)

        # Event trigger 1: Meaningful state transition
        if confirmed_behavior != state.get("last_event_behavior"):
            is_event = True
            state["last_event_behavior"] = confirmed_behavior
            state["last_event_time"] = current_time
        # Event trigger 2: Sustained HIGH/CRITICAL alert after cooldown
        elif risk_level in ["HIGH", "CRITICAL"] and time_since_last_event > self.config["risk_cooldown"]:
            is_event = True
            state["last_event_time"] = current_time

        timestamp_str = datetime.now().strftime("%H:%M:%S")
        event_summary = f"[{timestamp_str}] {species.capitalize()} #{track_id} -> {confirmed_behavior} -> {risk_level}"

        return {
            "track_id": track_id,
            "animal": species,
            "behavior": confirmed_behavior,
            "raw_candidate": raw_behavior,
            "confidence": round(confidence, 2),
            "speed": round(speed, 1),
            "direction": direction,
            "duration": duration,
            "risk_level": risk_level,
            "distance_travelled": round(cum_dist, 1),
            "human_distance": min_human_dist,
            "timestamp": datetime.now().isoformat(),
            "is_event": is_event,
            "event_summary": event_summary,
            "interacting_with": interacting_tracks,
            "loitering": is_loitering,
        }

    def cleanup_track(self, track_id):
        """Cleans up internal state for lost tracks."""
        if track_id in self.track_states:
            del self.track_states[track_id]
