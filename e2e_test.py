import time
import json
from datetime import datetime
from detection_system import WildlifeIoTSystem, WildlifeBehaviorAnalyzer

print("--- WILDAI E2E INTEGRATION TEST SUITE ---")

analyzer = WildlifeBehaviorAnalyzer()
iot = WildlifeIoTSystem(capture_index="test")

def run_test(name, species, boxes, track_history, prev_alert_time):
    print(f"\n[TEST] {name}")
    start = time.time()
    
    # 1. Behavior
    metrics = {}
    for box in boxes:
        sp = box['species']
        tid = box['tid']
        hist = track_history.get(tid, [])
        cx, cy = box['cx'], box['cy']
        
        b_metrics = analyzer.analyze(tid, hist, cx, cy, species, [(b['cx'], b['cy']) for b in boxes if b['species'] == 'person'])
        
        # 2. Risk
        risk_score, danger = iot.calculate_dynamic_risk(sp, b_metrics.get("distance_travelled", 0), b_metrics.get("speed", 0), b_metrics.get("human_distance"), box['conf'])
        
        # 3. Alert (simulate)
        if risk_score > 40:
            pass # trigger_alert would run here
            
        metrics[tid] = {"species": sp, "danger": danger, "risk": risk_score, "behavior": b_metrics.get("behavior", "unknown")}
        
    latency = (time.time() - start) * 1000
    
    print(f"INPUT: {len(boxes)} objects ({species})")
    for k, v in metrics.items():
        print(f"  Track #{k}: {v['species']} | Risk: {v['risk']} ({v['danger']}) | Behavior: {v['behavior']}")
    print(f"LATENCY: {latency:.2f}ms")

# Test 1: Human only
run_test("1. Human only", ["person"], [{'species': 'person', 'tid': 1, 'cx': 100, 'cy': 100, 'conf': 0.85}], {1: []}, 0)

# Test 3: Tiger only
run_test("3. Tiger", ["tiger"], [{'species': 'tiger', 'tid': 2, 'cx': 200, 'cy': 200, 'conf': 0.90}], {2: []}, 0)

# Test 4: Human + Tiger
run_test("4. Human + Tiger", ["tiger", "person"], [
    {'species': 'tiger', 'tid': 3, 'cx': 150, 'cy': 150, 'conf': 0.88},
    {'species': 'person', 'tid': 4, 'cx': 160, 'cy': 160, 'conf': 0.95}
], {3: [], 4: []}, 0)

# Test 6: Tiger approaching human
run_test("6. Animal approaching human", ["tiger", "person"], [
    {'species': 'tiger', 'tid': 5, 'cx': 110, 'cy': 110, 'conf': 0.88},
    {'species': 'person', 'tid': 6, 'cx': 100, 'cy': 100, 'conf': 0.95}
], {5: [(150, 150, time.time()-1, 1.0, 100)], 6: []}, 0)

# Test 8: Low confidence
run_test("8. Low confidence detection", ["tiger"], [{'species': 'tiger', 'tid': 7, 'cx': 200, 'cy': 200, 'conf': 0.20}], {7: []}, 0)

print("\nTests complete.")
