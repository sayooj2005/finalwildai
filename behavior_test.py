import sys, time
sys.path.insert(0, '.')
from behavior_analysis import WildlifeBehaviorAnalyzer

ba = WildlifeBehaviorAnalyzer()

# Build history as list of (cx, cy, timestamp) tuples - elephant approaching camera
base_time = time.time()
frames = [
    (300, 400, 100, 120),
    (305, 415, 108, 130),
    (312, 435, 118, 143),
    (322, 460, 130, 158),
    (335, 490, 145, 176),
]

history = [(cx, cy, base_time + i * 0.1, w, h) for i, (cx, cy, w, h) in enumerate(frames)]
last_bbox = frames[-1]  # (cx, cy, w, h)

result = ba.analyze(
    track_id=1,
    species='elephant',
    history=history,
    group_size=1,
    first_seen=base_time,
    active_humans=[],
    current_bbox=last_bbox,
    other_tracks=[],
    frame_shape=(720, 1280)
)

if result:
    print("PASS: Behavioral analysis working")
    print("  Species:   " + result["animal"])
    print("  Behavior:  " + result["behavior"])
    print("  Risk:      " + result["risk_level"])
    print("  Speed:     " + str(result["speed"]))
    print("  Direction: " + result["direction"])
    print("  Confidence:" + str(result["confidence"]))
    sys.exit(0)
else:
    print("FAIL: No result returned")
    sys.exit(1)
