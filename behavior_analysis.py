import math
import time
from datetime import datetime

class WildlifeBehaviorAnalyzer:
    """
    Independent module for wildlife behavior classification.
    Distinguishes OBSERVED BEHAVIOUR from RISK INTERPRETATION.
    Does not claim scientific validation; uses heuristics based on trajectory and proximity.
    """
    
    def __init__(self):
        # Configuration for behavior thresholds
        self.STATIONARY_SPEED_THRESHOLD = 15.0  # pixels/sec
        self.WALKING_SPEED_THRESHOLD = 80.0     # pixels/sec
        self.LINGERING_TIME_THRESHOLD = 5.0     # seconds

    def _calculate_distance(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def analyze(self, track_id, species, history, group_size, first_seen, active_humans):
        current_time = time.time()
        duration = round(current_time - first_seen, 1)
        
        # Default empty state
        if len(history) < 2:
            return {
                "species": species,
                "track_id": track_id,
                "behaviour": "stationary",
                "confidence": 0.5,
                "movement_direction": "none",
                "speed": 0.0,
                "distance_travelled": 0.0,
                "human_distance": None,
                "timestamp": datetime.now().isoformat()
            }

        # Current state
        cx, cy, t, w, h = history[-1]
        
        # Human Proximity
        min_human_dist = None
        for hx, hy in active_humans:
            dist = self._calculate_distance((cx, cy), (hx, hy))
            if min_human_dist is None or dist < min_human_dist:
                min_human_dist = round(dist, 1)

        # Distance Travelled & Speed
        distance_travelled = 0.0
        for i in range(1, len(history)):
            distance_travelled += self._calculate_distance((history[i-1][0], history[i-1][1]), (history[i][0], history[i][1]))
            
        dt = max(t - history[-2][2], 0.01)
        speed = self._calculate_distance((history[-2][0], history[-2][1]), (cx, cy)) / dt

        # Moving Direction
        dx = cx - history[-2][0]
        dy = cy - history[-2][1]
        
        if speed < self.STATIONARY_SPEED_THRESHOLD:
            direction = "stationary"
        elif abs(dx) > abs(dy):
            direction = "moving right" if dx > 0 else "moving left"
        else:
            direction = "moving away (up)" if dy < 0 else "approaching (down)"

        # Behaviour Classification State Machine
        behaviour = "stationary"
        confidence = 0.6
        
        if group_size >= 2 and speed > self.STATIONARY_SPEED_THRESHOLD:
            behaviour = "group movement"
            confidence = 0.8
        elif speed > self.WALKING_SPEED_THRESHOLD:
            behaviour = "running"
            confidence = 0.9
        elif speed > self.STATIONARY_SPEED_THRESHOLD:
            if dy > 20: # moving down towards camera significantly
                behaviour = "approaching"
                confidence = 0.85
            elif dy < -20:
                behaviour = "moving away"
                confidence = 0.85
            else:
                behaviour = "walking"
                confidence = 0.75
        else:
            if duration > self.LINGERING_TIME_THRESHOLD:
                behaviour = "lingering"
                confidence = 0.8
            else:
                behaviour = "stationary"
                confidence = 0.9

        return {
            "species": species,
            "track_id": track_id,
            "behaviour": behaviour,
            "confidence": confidence,
            "movement_direction": direction,
            "speed": round(speed, 1),
            "distance_travelled": round(distance_travelled, 1),
            "human_distance": min_human_dist,
            "timestamp": datetime.now().isoformat()
        }
