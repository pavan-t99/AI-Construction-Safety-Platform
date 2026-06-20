# safety_pipeline.py
import cv2
from ultralytics import YOLO
import csv
import os
from datetime import datetime, timedelta
import json
import time
from Incident_Analysis import GROQ_report
import tempfile
import sys

def run_ai_pipeline(video_source=0, camera_id="CAM_01",inference_size=640):
    print(f"🚀 Starting pipeline for Camera: {camera_id} | Source: {video_source}")
        # === PER-CAMERA STORAGE (Production Ready) ===
    base_dir = os.path.join("data", camera_id)
    # Create evidence folder
    os.makedirs(os.path.join(base_dir, "evidence"), exist_ok=True)
    
    violation_log_path = os.path.join(base_dir, "violation_log.csv")
    site_safety_path = os.path.join(base_dir, "Site_Safety.json")
    worker_history_path = os.path.join(base_dir, "worker_history.json")
    completed_incidents_path = os.path.join(base_dir, "completed_incidents_analysis.json")
    alerts_path = os.path.join(base_dir, "alerts.json")
    live_frame_path = os.path.join(base_dir, "live_frame.jpg")
    MODEL_PATH = "ppe_yolov8s_best.pt"
    COOLDOWN_SECONDS = 15
    MIN_VIOLATION_DURATION = 5

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Could not open source: {video_source}")
        print("🔄 Trying alternative webcam index (1)...")
        cap = cv2.VideoCapture(0)   # Try index 1
        
        if not cap.isOpened():
            print("❌ No webcam found. Please check your camera connection.")
            print("💡 Try closing other apps using the camera (Zoom, Teams, etc.)")
            return
        else:
            print("✅ Opened webcam with index 1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not os.path.exists(violation_log_path):
        with open(violation_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["TIMESTAMP", "PERSON_ID", "VIOLATION", "CONFIDENCE", "DURATION", "IMAGE_PATH"])
        
        # Cleanup old evidence images (keep last 1000)
    evidence_dir = os.path.join(base_dir, "evidence")
    evidence_files = sorted([f for f in os.listdir(evidence_dir) if f.endswith('.jpg')],
                        key=lambda x: os.path.getctime(os.path.join(evidence_dir, x)))
        
    if len(evidence_files) > 1000:
        for old_file in evidence_files[:-800]:   # Keep latest 800
            try:
                os.remove(os.path.join(evidence_dir, old_file))
            except:
                pass
    # Original variables
    site_risk_score = 0
    site_risk_level = "SAFE"
    historical_site_risk_score = 0
    total_incidents = 0
    active_violations = {}
    completed_incidents_analysis = []
    worker_history = {}
    violation_alert_levels = {}   # (person_id, violation_type) -> (last_alert_time, level)
    violation_messages = {
        "NO-Gloves": "GLOVE VIOLATION", "NO-Goggles": "GOGGLE VIOLATION",
        "NO-Hardhat": "HELMET VIOLATION", "NO-Mask": "MASK VIOLATION",
        "NO-Safety Vest": "VEST VIOLATION"
    }

    severity_rules = {"HELMET VIOLATION": "HIGH", "MASK VIOLATION": "MEDIUM",
                      "GOGGLE VIOLATION": "MEDIUM", "VEST VIOLATION": "LOW",
                      "GLOVE VIOLATION": "LOW"}

    risk_points = {"HELMET VIOLATION": 30, "MASK VIOLATION": 15,
                   "GOGGLE VIOLATION": 20, "VEST VIOLATION": 20,
                   "GLOVE VIOLATION": 15}

    machinery_classes = ["Excavator", "Forklift", "Vehicle", "Crane", "Bulldozer"]

    model = YOLO(MODEL_PATH)
    violation_cooldowns = {}
    prev_time = time.time()
    last_frame_save = time.time()
    last_json_update = time.time()
    person_last_seen = {}
    attempts=0
    PERSON_TIMEOUT = 30  # seconds
    print("🚀 Pipeline Started - Press 'q' to stop")

    try:
        while True:
            ret, frame = cap.read()
            if not ret and (attempts<10):
                print("⚠️ Camera disconnected. Attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(video_source)
                attempts += 1
                continue   # Try again

            # === CRITICAL: SCALING FOR PERFECT ALIGNMENT ===
            orig_h, orig_w = frame.shape[:2]
            scale_x = orig_w / inference_size
            scale_y = orig_h / inference_size

            inference_frame = cv2.resize(frame, (inference_size, inference_size))
            results = model.track(inference_frame, persist=True, imgsz=inference_size)

            current_time = datetime.now()
            detected_people = {}
            detected_violations = []
            detected_machinery = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                boxes = results[0].boxes

                for box, track_id in zip(boxes, track_ids):
                    # Scale back to original resolution
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    xmin = int(x1 * scale_x)
                    ymin = int(y1 * scale_y)
                    xmax = int(x2 * scale_x)
                    ymax = int(y2 * scale_y)

                    cls_name = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])

                    if cls_name == "Person":
                        if confidence <= 0.7: continue
                        if (xmax - xmin) > (ymax - ymin): continue
                        # Simple re-identification logic
                        current_time_sec = time.time()
                        # Clean old track IDs (simple re-identification)
                        for old_id in list(person_last_seen.keys()):
                            if current_time_sec - person_last_seen[old_id] > PERSON_TIMEOUT:
                                if old_id in worker_history:
                                    del person_last_seen[old_id]
                        person_last_seen[track_id] = current_time_sec
                        detected_people[track_id] = {
                            "bbox": (xmin, ymin, xmax, ymax),
                            "confidence": confidence,
                            "has_violation": False,
                            "violated_duration": timedelta(0),
                            "violations": []
                        }
                    elif cls_name in violation_messages:
                        detected_violations.append({
                            "cls_name": cls_name,
                            "violation_type": violation_messages[cls_name],
                            "confidence": confidence,
                            "center": ((xmin + xmax) // 2, (ymin + ymax) // 2)
                        })
                    elif cls_name in machinery_classes:
                        detected_machinery.append({"bbox": (xmin, ymin, xmax, ymax)})

                # === YOUR ORIGINAL ASSOCIATION LOGIC ===
                for violation in detected_violations:
                    v_x, v_y = violation["center"]
                    for p_id, p_data in detected_people.items():
                        p_xmin, p_ymin, p_xmax, p_ymax = p_data["bbox"]
                        if p_xmin <= v_x <= p_xmax and p_ymin <= v_y <= p_ymax:
                            dict_key = (p_id, violation["violation_type"])
                            if dict_key not in active_violations:
                                active_violations[dict_key] = {
                                    "person_id": p_id, "start_time": current_time,
                                    "evidence_captured": False, "last_seen": current_time,
                                    "confidence": 0, "violation_type": violation["violation_type"]
                                }
                            else:
                                active_violations[dict_key]["last_seen"] = current_time

                            duration = current_time - active_violations[dict_key]["start_time"]
                            p_data["has_violation"] = True
                            p_data["violations"].append(violation)

                            if duration > timedelta(seconds=MIN_VIOLATION_DURATION):
                                p_data["violated_duration"] = duration
                                cropped_frame = frame[p_ymin:p_ymax, p_xmin:p_xmax]                               
                                # === PER-CAMERA IMAGE SAVING ===
                                file_name = f"{violation['violation_type']}_Person-{p_id}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                                full_image_path = os.path.join(base_dir, "evidence", file_name)
                                if violation["confidence"] > active_violations[dict_key].get("confidence", 0):
                                    cv2.imwrite(full_image_path, cropped_frame)
                                    active_violations[dict_key]["confidence"] = violation["confidence"]
                                    active_violations[dict_key]["Image_path"] = full_image_path   # Full path for Streamlit


                                active_violations[dict_key]["duration"] = duration
                                active_violations[dict_key]["risk_score"] = risk_points[violation["violation_type"]]
                                active_violations[dict_key]["risk_level"] = severity_rules[violation["violation_type"]]
                                active_violations[dict_key]["evidence_captured"] = True

                                # === SMART ALERT + COOLDOWN LOGIC ===
                                                                
                                alert_key = (p_id, violation["violation_type"])
                                
                                # Initialize alert tracking if first time
                                if alert_key not in violation_alert_levels:
                                    violation_alert_levels[alert_key] = {
                                        "last_alert": current_time,
                                        "level": 1
                                    }
                                    send_alert = True
                                else:
                                    time_since_last = (current_time - violation_alert_levels[alert_key]["last_alert"]).total_seconds()
                                    level = violation_alert_levels[alert_key]["level"]
                                    send_alert = False

                                    if level == 1 and time_since_last > 30:           # First reminder after 30 sec
                                        violation_alert_levels[alert_key]["level"] = 2
                                        violation_alert_levels[alert_key]["last_alert"] = current_time
                                        send_alert = True
                                    elif level == 2 and time_since_last > 300:        # Escalation after 5 minutes
                                        violation_alert_levels[alert_key]["level"] = 3
                                        violation_alert_levels[alert_key]["last_alert"] = current_time
                                        send_alert = True

                                # Update cooldown for logging (keep your original COOLDOWN_SECONDS)
                                if dict_key not in violation_cooldowns or current_time >= violation_cooldowns[dict_key] + timedelta(seconds=COOLDOWN_SECONDS):
                                    violation_cooldowns[dict_key] = current_time

                                    # Log to CSV only when we decide to "act"
                                    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                    with open(violation_log_path, "a", newline="") as f:
                                        writer = csv.writer(f)
                                        writer.writerow([timestamp_str, f"Person-{p_id}", violation["violation_type"], 
                                                        f"{violation['confidence']:.2f}", str(duration), full_image_path])

                                # Send Alert (Telegram / Print / Future Notification)
                                if send_alert:
                                    alert_msg = f"🚨 {'ESCALATION' if violation_alert_levels[alert_key]['level'] == 3 else 'REMINDER' if violation_alert_levels[alert_key]['level'] == 2 else 'ALERT'}: {violation['violation_type']} - Person-{p_id}"
                                    print(alert_msg)
                                    # Save to alert history
                                    alert_record = {
                                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "camera_id": camera_id,
                                        "person_id": p_id,
                                        "alert_level": violation_alert_levels[alert_key]["level"],
                                        "violation_type": violation["violation_type"],
                                        "message": alert_msg
                                    }
                                    try:
                                        # FIX: Use alerts_path instead of the global "alerts.json" string
                                        if os.path.exists(alerts_path):
                                            with open(alerts_path, "r") as f:
                                                alerts = json.load(f)
                                        else:
                                            alerts = []
                                        alerts.append(alert_record)
                                        if len(alerts) > 1000:
                                            alerts = alerts[-800:]
                                        with open(alerts_path, "w") as f:
                                            json.dump(alerts, f, indent=4, default=str)
                                    except:
                                        pass
                                    # TODO: Add Telegram notification here later

                # Machinery Proximity
                for p_id, p_data in detected_people.items():
                    p_xmin, p_ymin, p_xmax, p_ymax = p_data["bbox"]
                    for machine in detected_machinery:
                        person_center_x = (p_xmin + p_xmax) // 2
                        person_center_y = (p_ymin + p_ymax) // 2
                        mx1, my1, mx2, my2 = machine["bbox"]
                        machine_center_x = (mx1 + mx2) // 2
                        machine_center_y = (my1 + my2) // 2
                        distance = ((person_center_x - machine_center_x) ** 2 + 
                                   (person_center_y - machine_center_y) ** 2) ** 0.5
                        if distance < 200:
                            for key in list(active_violations.keys()):
                                if key[0] == p_id:
                                    active_violations[key]["near_machinery"] = True

            # === VISUAL RENDERING (Always correct coordinates) ===
            for p_id, p_data in detected_people.items():
                px1, py1, px2, py2 = p_data["bbox"]
                if p_data["has_violation"]:
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                    v_labels = ", ".join([v["cls_name"] for v in p_data["violations"]])
                    cv2.putText(frame, f"Person-{p_id} | {v_labels}", (px1, py1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"RISK: {p_data.get('risk_level', 'LOW')}", (px1, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person-{p_id}: Safe Passthrough", (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # FPS + Site Risk
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"SITE RISK: {site_risk_level} ({site_risk_score})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Advanced Association Production Pipeline", frame)

            # === RELIABLE LIVE FRAME SAVE FOR STREAMLIT ===
            if time.time() - last_frame_save > 2.5:   # Every 2.5 seconds
                try:
                    # Optional: resize for faster save & lower CPU
                    display_frame = cv2.resize(frame, (1280, 720))
                    cv2.imwrite(live_frame_path, display_frame)
                    last_frame_save = time.time()
                except Exception as e:
                    print(f"Frame save warning: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # === FULL ORIGINAL INCIDENT + GROQ + JSON LOGIC ===
            for dict_key in list(active_violations.keys()):
                last_seen = active_violations[dict_key]["last_seen"]
                if current_time - last_seen > timedelta(seconds=3):
                    if active_violations[dict_key]["evidence_captured"]:
                        start_time = active_violations[dict_key]["start_time"]
                        incident = {
                            "person_id": active_violations[dict_key]["person_id"],
                            "violation_type": active_violations[dict_key]["violation_type"],
                            "start_time": str(start_time),
                            "end_time": str(last_seen),
                            "duration_seconds": (last_seen - start_time).total_seconds(),
                            "confidence": active_violations[dict_key].get("confidence", 0),
                            "Image_path": active_violations[dict_key].get("Image_path", "")
                        }
                        incident["severity"] = severity_rules[incident["violation_type"]]
                        incident["risk_score"] = risk_points[incident["violation_type"]]

                        if active_violations[dict_key].get("near_machinery", False):
                            incident["risk_score"] += 20
                        same_person_count = sum(1 for k in active_violations if k[0] == incident["person_id"])
                        if same_person_count > 1:
                            incident["risk_score"] += 20
                        seconds = (last_seen - start_time).total_seconds()
                        incident["risk_score"] += int(seconds // 30) * 10

                        if incident["risk_score"] < 30:
                            incident["risk_level"] = "LOW"
                        elif incident["risk_score"] < 70:
                            incident["risk_level"] = "MEDIUM"
                        else:
                            incident["risk_level"] = "HIGH"

                        historical_site_risk_score += incident["risk_score"]
                        total_incidents += 1
                        incident["GROQ_analysis"] = GROQ_report(incident)
                        completed_incidents_analysis.append(incident)
                        if len(completed_incidents_analysis) > 500:
                            completed_incidents_analysis = completed_incidents_analysis[-400:]  # Keep last 400
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                            json.dump(completed_incidents_analysis, tmp, indent=4, default=str)
                            tmp_path = tmp.name

                        os.replace(tmp_path, completed_incidents_path)
                        # Worker History (your original code)
                        p_id = incident["person_id"]
                        if p_id not in worker_history:
                            worker_history[p_id] = {"person_id": p_id, "risk_score": 0, "violations": set(), "incidents": []}
                        worker_history[p_id]["risk_score"] += incident["risk_score"]
                        worker_history[p_id]["violations"].add(incident["violation_type"])
                        worker_history[p_id]["incidents"].append(incident)
                        if len(worker_history[p_id]["incidents"]) > 50:
                            worker_history[p_id]["incidents"] = worker_history[p_id]["incidents"][-30:]
                        workers = []
                        for worker in worker_history.values():
                            workers.append({
                                "person_id": worker["person_id"],
                                "risk_score": worker["risk_score"],
                                "risk_level": "LOW" if worker["risk_score"] < 50 else "MEDIUM" if worker["risk_score"] < 100 else "HIGH",
                                "violations": list(worker["violations"]),
                                "unique_violation_count": len(worker["violations"]),
                                "incidents": worker["incidents"]
                            })
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                            json.dump(workers, tmp, indent=4, default=str)
                            tmp_path = tmp.name

                        os.replace(tmp_path, worker_history_path)
                    del active_violations[dict_key]

            # Site Safety
            # site_risk_score = sum(v.get("risk_score", 0) for v in active_violations.values())
            # site_risk_level = "SAFE" if site_risk_score == 0 else "WARNING" if site_risk_score < 50 else "CRITICAL"

            # site_safety = {
            #     "current_site_risk_score": site_risk_score,
            #     "current_site_risk_level": site_risk_level,
            #     "historical_site_risk_score": historical_site_risk_score,
            #     "total_incidents": total_incidents,
            #     "active_violations": len(active_violations),
            #     "workers_with_history": len(worker_history)
            # }
            # with open("Site_Safety.json", "w") as f:
            #     json.dump(site_safety, f, indent=4, default=str)

                        # === Site Safety Update (Throttled) ===
            site_risk_score = sum(v.get("risk_score", 0) for v in active_violations.values())
            site_risk_level = "SAFE" if site_risk_score == 0 else "WARNING" if site_risk_score < 50 else "CRITICAL"

            # Throttle JSON writes
            if time.time() - last_json_update > 2.0:
                site_safety = {
                    "current_site_risk_score": site_risk_score,
                    "current_site_risk_level": site_risk_level,
                    "historical_site_risk_score": historical_site_risk_score,
                    "total_incidents": total_incidents,
                    "active_violations": len(active_violations),
                    "active_workers_in_violation": len({k[0] for k in active_violations.keys()}),  # New: unique workers
                    "workers_with_history": len(worker_history)
                }               
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                    json.dump(site_safety, tmp, indent=4, default=str)
                    tmp_path = tmp.name
                os.replace(tmp_path, site_safety_path)
                last_json_update = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if isinstance(video_source, str) and "temp" in video_source.lower() and os.path.exists(video_source):
            try:
                os.unlink(video_source)
            except:
                pass
        print("✅ Pipeline stopped gracefully.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        cam_id = sys.argv[2] if len(sys.argv) > 2 else "CAM_01"
        run_ai_pipeline(video_source=source if source.isdigit() else source, camera_id=cam_id,inference_size=640)
    else:
        run_ai_pipeline(video_source=0, camera_id="CAM_01",inference_size=640)