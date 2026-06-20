import cv2
import csv
import os
from datetime import datetime, timedelta
from ultralytics import YOLO

# 1. Initialize your custom fine-tuned model
model = YOLO("ppe_yolov8s_best.pt")

# 2. Setup your industrial violation map and file infrastructure
if not os.path.exists("violation_log.csv"):
    with open("violation_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TIMESTAMP", "PERSON_ID", "VIOLATION_STATUS", "CONFIDENCE"])

violation_messages = {
    "NO-Gloves": "GLOVE VIOLATION",
    "NO-Goggles": "GOGGLE VIOLATION",
    "NO-Hardhat": "HELMET VIOLATION",
    "NO-Mask": "MASK VIOLATION",
    "NO-Safety Vest": "VEST VIOLATION"
}

# 3. Memory Layer State Configuration
violation_cooldowns = {}
COOLDOWN_SECONDS = 15  # Log a specific person's violation at most once every 15 seconds

# Bind to webcam (0) or path to a test video file
cap = cv2.VideoCapture(0)

print("🚀 Fully Associated Multi-Person Safety System Active...")
violation_found = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the tracking engine
    results = model.track(source=frame, persist=True, conf=0.25, verbose=False)
    boxes = results[0].boxes
    current_time = datetime.now()

    # Dictionaries to separate people and items detected in this specific frame
    detected_people = {}
    detected_violations = []

    # --- STEP 1: PARSE & TRACK INDIVIDUALS IN THE FRAME ---
    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if cls_name == "Person":
                # Store person data for spatial association check
                detected_people[track_id] = {
                    "bbox": (xmin, ymin, xmax, ymax),
                    "confidence": confidence,
                    "has_violation": False,
                    "violations": []
                }
            elif cls_name in violation_messages:
                # Keep track of active safety gear violations in this frame
                violation_found = True
                file_name
                detected_violations.append({
                    "cls_name": cls_name,
                    "violation_type": violation_messages[cls_name],
                    "confidence": confidence,
                    "center": ((xmin + xmax) // 2, (ymin + ymax) // 2) # Geometric center point of the violation box
                })

        # --- STEP 2: ADVANCED SPATIAL ASSOCIATION LOGIC ---
        # Look at every violation box found and check which Person box contains its center point
        for violation in detected_violations:
            v_x, v_y = violation["center"]
            associated = False
            
            for p_id, p_data in detected_people.items():
                p_xmin, p_ymin, p_xmax, p_ymax = p_data["bbox"]
                
                # Check if the violation center point lies inside this specific person's box boundaries
                if p_xmin <= v_x <= p_xmax and p_ymin <= v_y <= p_ymax:
                    p_data["has_violation"] = True
                    p_data["violations"].append(violation)
                    associated = True
                    
                    # --- STEP 3: APPLY THE MEMORY LAYER COOLDOWN CONTROL ---
                    dict_key = (p_id, violation["violation_type"])
                    if dict_key in violation_cooldowns:
                        if current_time < violation_cooldowns[dict_key] + timedelta(seconds=COOLDOWN_SECONDS):
                            continue # Still inside cooldown period, skip duplicate logging
                    
                    # Passed cooldown: Update memory timer and write official log entry
                    violation_cooldowns[dict_key] = current_time
                    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    with open("violation_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_str, f"Person-{p_id}", violation["violation_type"], f"{violation['confidence']:.2f}"])
                    
                    print(f"⚠️ [ALERT LOGGED] Person-{p_id} committed {violation['violation_type']}")

        # --- STEP 4: VISUAL DASHBOARD RENDERING ---
        for p_id, p_data in detected_people.items():
            px1, py1, px2, py2 = p_data["bbox"]
            
            if p_data["has_violation"]:
                # Draw a warning red box around safety rule violators
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                v_labels = ", ".join([v["cls_name"] for v in p_data["violations"]])
                cv2.putText(frame, f"Person-{p_id}: VIOLATION ({v_labels})", (px1, py1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Draw a secure green box around compliant workers
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person-{p_id}: Safe Passthrough", (px1, py1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the production interface frame live
    if violation_found:
        cv2.imwrite(file_name, frame)
    #cv2.imshow("Custom Detection", frame)
    cv2.imshow("Advanced Association Production Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔒 System shutdown cleanly.")