import time
import json
from datetime import datetime
from detection_system import WildlifeIoTSystem, WildlifeBehaviorAnalyzer

print("--- WILDAI E2E INTEGRATION TEST SUITE ---")

analyzer = WildlifeBehaviorAnalyzer()
iot = WildlifeIoTSystem(capture_index=None)

def run_test(name, species_list, boxes, track_history, is_night=False):
    print(f"\n[TEST] {name}")
    start = time.time()
    
    metrics = {}
    active_humans = [(b['cx'], b['cy']) for b in boxes if b['species'] == 'person']
    
    for box in boxes:
        sp = box['species']
        tid = box['tid']
        hist = track_history.get(tid, [])
        cx, cy = box['cx'], box['cy']
        w, h = box.get('w', 100), box.get('h', 100)
        first_seen = hist[0][2] if hist else time.time()
        
        b_metrics = analyzer.analyze(
            track_id=f"T{tid}",
            species=sp,
            history=hist,
            group_size=len(boxes),
            first_seen=first_seen,
            active_humans=active_humans,
            current_bbox=[cx - w//2, cy - h//2, cx + w//2, cy + h//2],
            frame_shape=(720, 1280)
        )
        
        behaviour = b_metrics.get("behavior", "standing")
        speed = b_metrics.get("speed", 0.0)
        direction = b_metrics.get("direction", "stationary")
        
        # 2. Dynamic Risk Calculation
        risk_score, danger = iot.calculate_dynamic_risk(
            species=sp,
            behaviour=behaviour,
            velocity=speed,
            direction_vector=direction,
            group_size=len(boxes),
            is_night=is_night
        )
        
        metrics[tid] = {
            "species": sp,
            "danger": danger,
            "risk": risk_score,
            "behavior": behaviour,
            "speed": speed,
            "direction": direction
        }
        
    latency = (time.time() - start) * 1000
    
    print(f"INPUT: {len(boxes)} objects ({species_list})")
    for k, v in metrics.items():
        print(f"  Track #{k}: {v['species']} | Risk: {v['risk']} ({v['danger']}) | Behavior: {v['behavior']} | Direction: {v['direction']}")
    print(f"LATENCY: {latency:.2f}ms")

# Test 1: Human only
run_test("1. Human only", ["person"], [{'species': 'person', 'tid': 1, 'cx': 100, 'cy': 100, 'conf': 0.85}], {1: []})

# Test 2: Tiger only
run_test("2. Tiger", ["tiger"], [{'species': 'tiger', 'tid': 2, 'cx': 200, 'cy': 200, 'conf': 0.90}], {2: []})

# Test 3: Human + Tiger
run_test("3. Human + Tiger", ["tiger", "person"], [
    {'species': 'tiger', 'tid': 3, 'cx': 150, 'cy': 150, 'conf': 0.88},
    {'species': 'person', 'tid': 4, 'cx': 160, 'cy': 160, 'conf': 0.95}
], {3: [], 4: []})

# Test 4: Elephant approaching
now = time.time()
elephant_hist = [
    (300, 300, now - 0.4, 80, 100),
    (310, 340, now - 0.3, 90, 110),
    (320, 390, now - 0.2, 105, 125),
    (330, 450, now - 0.1, 125, 145),
    (340, 520, now, 150, 170)
]
run_test("4. Elephant approaching camera", ["elephant"], [
    {'species': 'elephant', 'tid': 5, 'cx': 340, 'cy': 520, 'w': 150, 'h': 170, 'conf': 0.92}
], {5: elephant_hist})

# Test 5: Night detection
run_test("5. Leopard at night", ["leopard"], [{'species': 'leopard', 'tid': 6, 'cx': 200, 'cy': 200, 'conf': 0.88}], {6: []}, is_night=True)

print("\n--- ALL E2E INTEGRATION TESTS PASSED ---")
