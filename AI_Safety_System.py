import cv2
from ultralytics import YOLO
import csv
import os
from datetime import datetime , timedelta
import json
from Incident_Analysis import GROQ_report
if not os.path.exists("violation_log.csv"):
    with open("violation_log.csv", "a",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["TIMESTAMP","VIOLATION","CONFIDENCE"])

site_risk_score = 0
site_risk_level = "SAFE"
historical_site_risk_score = 0
total_incidents = 0
active_violations = {}
completed_incidents = []
completed_incidents_analysis = []
violation_messages={"NO-Gloves":"GLOVE VIOLATION",
                    "NO-Goggles":"GOGGLE VIOLATION",
                    "NO-Hardhat":"HELMET VIOLATION",
                    "NO-Mask":"MASK VIOLATION",
                    "NO-Safety Vest":"VEST VIOLATION"}

severity_rules = {
    "HELMET VIOLATION": "HIGH",
    "MASK VIOLATION": "MEDIUM",
    "GOGGLE VIOLATION": "MEDIUM",
    "VEST VIOLATION": "LOW",
    "GLOVE VIOLATION": "LOW"
}

risk_points = {
    "HELMET VIOLATION": 30,
    "MASK VIOLATION": 15,
    "GOGGLE VIOLATION": 20,
    "VEST VIOLATION": 20,
    "GLOVE VIOLATION": 15
}

machinery_classes = [
    "Excavator",
    "Forklift",
    "Vehicle"
]

worker_history={}

#image_path= r"C:\Users\Anuj\Documents\AI_Safety_System\train\images\-424-_png_jpg.rf.9f5c737706f9c6166fefe5d5aa5d292c.jpg"
model=YOLO("ppe_yolov8s_best.pt")

violation_cooldowns = {}
COOLDOWN_SECONDS = 15

cap = cv2.VideoCapture(0)

#print(model.names)

violation_found = False
while True:
    ret,frame = cap.read()
    
    if not ret:
        print("FAILED to capture the frame")
        break

    results = model.track(frame, persist=True)
    current_time = datetime.now()
    #crating dict to seperate people and detecting the violating boxes coords in the same frame
    detected_people = {}
    detected_violations = []
    detected_machinery = []
    if results[0].boxes is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes
        act_violations = {}
        for box, track_id in zip(boxes, track_ids):
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                cls_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])               
                if cls_name == "Person":
                    # Store person data for spatial association check
                    if confidence<=0.7:
                        continue
                    # FILTER B: Aspect Ratio Check (Humans are vertically oriented boxes)
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    # If a box is wider than it is tall (like a quadraped animal), reject it
                    if box_width > box_height:
                        continue  # Drops horizontal animal shapes
                    detected_people[track_id] = {
                        "bbox": (xmin, ymin, xmax, ymax),
                        "confidence": confidence,
                        "has_violation": False,
                        "violated_duration" : timedelta(0),
                        "violations": []
                    }
                elif cls_name in violation_messages:
                    # Keep track of active safety gear violations in this frame
                    
                    violation_found = True
                    file_name = f"{violation_messages[cls_name]}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                    detected_violations.append({
                        "cls_name": cls_name,
                        "violation_type": violation_messages[cls_name],
                        "confidence": confidence,
                        "file_name": f"{violation_messages[cls_name]}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg",
                        "center": ((xmin + xmax) // 2, (ymin + ymax) // 2) # Geometric center point of the violation box
                    })
                
                elif cls_name in machinery_classes:
                    detected_machinery.append({"bbox": (xmin,ymin,xmax,ymax)})

                #act_violations[detected_people[track_id] , detected_violations["violation_type"]] = { "start_time": current_time ,
                                                                                                     #   "evidence_captured": False}
        # --- STEP 2: ASSOCIATION OF VIOLATIONS TO different people ---
        for violation in detected_violations:
            v_x, v_y = violation["center"]
            associated = False
            current_score = risk_points[violation["violation_type"]]
            current_level = severity_rules[violation["violation_type"]]
            for p_id, p_data in detected_people.items():
                    p_xmin, p_ymin, p_xmax, p_ymax = p_data["bbox"]
                    
                    # Check if the violation center point lies inside this specific person's box boundaries
                    if p_xmin <= v_x <= p_xmax and p_ymin <= v_y <= p_ymax:
                        associated = True
                        dict_key = (p_id, violation["violation_type"])
                        if dict_key not in active_violations:
                            active_violations[dict_key] = {
                                "person_id": p_id,
                                "start_time": current_time,
                                "evidence_captured": False ,
                                "last_seen": current_time,
                                "confidence": 0,
                                "violation_type": violation["violation_type"]
                                                               
                            }
                        else:
                            active_violations[dict_key]["last_seen"] = current_time                           
                        
                        duration = current_time - active_violations[dict_key]["start_time"]
                        p_data["has_violation"] = True
                        p_data["violations"].append(violation)
                         # violation is detected
                        risk_score = active_violations.get(dict_key, {}).get("risk_score", 0)
                        risk_level = active_violations.get(dict_key, {}).get("risk_level", "LOW")
                        if duration > timedelta(seconds=5): 
                            p_data["violated_duration"] = duration
                            cropped_frame = frame[p_ymin:p_ymax, p_xmin:p_xmax]
                            file_name = f"{violation['violation_type']}_Person-{p_id}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                            
                            # Save image ONLY when confidence improves
                            if violation["confidence"] > active_violations[dict_key].get("confidence", 0):

                                cv2.imwrite(file_name, cropped_frame)

                                active_violations[dict_key]["confidence"] = violation["confidence"]
                                active_violations[dict_key]["Image_path"] = file_name

                            active_violations[dict_key]["duration"] = duration
                            active_violations[dict_key]["violation_type"] = violation["violation_type"]
                            active_violations[dict_key]["risk_score"] = current_score
                            active_violations[dict_key]["risk_level"] = current_level
                            
                            if not active_violations[dict_key]["evidence_captured"]:
                                active_violations[dict_key]["evidence_captured"] = True
                            
                                                    
                            # APPLYING THE MEMORY LAYER COOLDOWN CONTROL [TRACKING PERSONS AND WAITING FOR 15 SECONDS] ---
                            
                            if dict_key in violation_cooldowns:
                                if current_time < violation_cooldowns[dict_key] + timedelta(seconds=COOLDOWN_SECONDS):
                                    continue # Still inside cooldown period, skip duplicate logging
                            
                            # Passed cooldown: Update memory timer and write official log entry
                            violation_cooldowns[dict_key] = current_time
                            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                            with open("violation_log.csv", "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([timestamp_str, f"Person-{p_id}", violation["violation_type"], f"{violation['confidence']:.2f}",f"duration: {p_data['violated_duration']}",f"Image_path: {file_name}"])
                        
                            print(f"<<< [ALERT LOGGED] Person-{p_id} committed {violation['violation_type']} >>>")

                        
                        print(active_violations)
        
                    near_machinery = False
                    for machine in detected_machinery:  
                        person_center_x = (p_xmin + p_xmax) // 2
                        person_center_y = (p_ymin + p_ymax) // 2
                        mx1,my1,mx2,my2 = machine["bbox"]
                        machine_center_x = (mx1 + mx2) // 2
                        machine_center_y = (my1 + my2) // 2
                        distance = ((person_center_x-machine_center_x)**2 +
                                    (person_center_y-machine_center_y)**2) ** 0.5
                        if distance < 200:
                            near_machinery = True
                            active_violations[dict_key]["near_machinery"] = near_machinery
                    # VISUAL DASHBOARD RENDERING ---
        for p_id, p_data in detected_people.items():
            px1, py1, px2, py2 = p_data["bbox"]
            if p_data["has_violation"]:
                # Drawing a warning red box around safety rule violators
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                v_labels = ", ".join([v["cls_name"] for v in p_data["violations"]])
                
                cv2.putText(frame,f"Person-{p_id} | {v_labels}",(px1, py1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255) , 2)

                cv2.putText(frame,f"RISK: {risk_level} ({risk_score})",(px1, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255), 2)
            else:
                # Drawing a secure green box around compliant workers
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame , f"Person-{p_id}: Safe Passthrough", (px1, py1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Capturing the violation frame and showing the visual dashboard
    # if violation_found and file_name != "":
    #     cv2.imwrite(file_name, frame)
    # REMOVING ACTIVE VIOLATIONS FROM ACTIVE TRACKING (If a violation disappears for more than 3 seconds, we consider it ended and remove it from active tracking)
    for dict_key in list(active_violations.keys()):
        last_seen = active_violations[dict_key]["last_seen"]
        if current_time - last_seen > timedelta(seconds=3):
            print(f"{dict_key} violation ended")
            
            if active_violations[dict_key]["evidence_captured"]:
                start_time = active_violations[dict_key]["start_time"]  
                incident = {
                        "person_id": active_violations[dict_key]["person_id"],
                        "violation_type": active_violations[dict_key]["violation_type"],
                        "start_time": start_time,
                        "end_time": last_seen,
                        "duration": last_seen - start_time,
                        "confidence":active_violations[dict_key].get("confidence", 0),
                        "Image_path": active_violations[dict_key].get("Image_path", "")
                            }
                incident["severity"] = severity_rules[incident["violation_type"]]
                incident["risk_score"] = risk_points[incident["violation_type"]]
                same_person_count = 0
                if active_violations[dict_key].get("near_machinery", False):
                    incident["risk_score"] += 20
                for key in active_violations:
                    if key[0] == active_violations[dict_key]["person_id"]:
                        same_person_count += 1
                if same_person_count > 1:
                    incident["risk_score"] += 20
                #is a person detects violating more than 30 sec ==>risk score will increase by 10 points for every 30 seconds of violation duration
                seconds = active_violations[dict_key]["duration"].total_seconds()
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
                with open("completed_incidents_analysis.json", "w") as f:
                    json.dump(completed_incidents_analysis, f, indent=4, default=str)

                p_id = incident["person_id"]

                if p_id not in worker_history:
                    worker_history[p_id] = {"person_id": p_id,
                                            "risk_score": 0,
                                            "violations": set(),
                                            "incidents": []}

                worker_history[p_id]["risk_score"] += incident["risk_score"]
                worker_history[p_id]["violations"].add(incident["violation_type"])
                worker_history[p_id]["incidents"].append(incident)
                workers = []

                for worker in worker_history.values():

                    workers.append({"person_id": worker["person_id"],
                        "risk_score": worker["risk_score"],
                        "risk_level":"LOW" if worker["risk_score"] < 50 else
                                    "MEDIUM" if worker["risk_score"] < 100 else
                                    "HIGH",
                        "violations": list(worker["violations"]),
                        "unique_violation_count": len(worker["violations"]),

                        "incidents": worker["incidents"]})
                
                with open("worker_history.json","w") as f:
                    json.dump(workers,f,indent=4,default=str)
                #site_risk_score += incident["risk_score"]
                
            del active_violations[dict_key]    
    

    site_risk_score = 0

    for violation_data in active_violations.values():
        site_risk_score += violation_data.get("risk_score", 0)
    if site_risk_score == 0:
        site_risk_level = "SAFE"

    elif site_risk_score < 50:
        site_risk_level = "WARNING"

    else:
        site_risk_level = "CRITICAL"

    site_safety = {
    "current_site_risk_score": site_risk_score,
    "current_site_risk_level": site_risk_level,

    "historical_site_risk_score": historical_site_risk_score,
    "total_incidents": total_incidents,

    "active_violations": len(active_violations),
    "workers_with_history": len(worker_history)
}

    with open("Site_Safety.json","w") as f:
        json.dump(site_safety,f,indent=4,default=str)
    cv2.putText(frame,f"SITE RISK: {site_risk_level} ({site_risk_score})",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("Advanced Association Production Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()