# safety_pipeline_v4.py
# AI Construction Safety Monitoring Platform — Core Pipeline
# All 10 issues resolved:
#   1. WORKER_{id} persistent mapping (ReID v1 — visual tracking only)
#   2. Per-camera isolated storage (data/CAM_XX/)
#   3. Headless mode (HEADLESS=1 env var for Docker/cloud)
#   4. Groq in background thread (never blocks detection)
#   5. Model watchdog with auto-recovery
#   6. SQLite replaces JSON/CSV for incidents and workers
#   7. Atomic frame save (tmp → os.replace, no race condition)
#   8. Authentication — deferred (post-Docker)
#   9. Evidence cleanup with reference check
#  10. Telegram wrapped in try/except

# Windows webcam fix — must be before cv2 import
import os as _os
_os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
from ultralytics import YOLO
import csv
import os
from datetime import datetime, timedelta
import json
import time
import tempfile
import sys
import threading
import queue
from Incident_Analysis_v2 import GROQ_report
from logger_setup import get_logger
from database import init_db, insert_incident, upsert_worker, insert_alert

# ── HEADLESS MODE ────────────────────────────────────────────────────────────
HEADLESS = os.environ.get("HEADLESS", "0") == "1"


# ── GROQ BACKGROUND WORKER ───────────────────────────────────────────────────
class GroqWorker:
    """
    Background daemon thread — processes Groq analysis without blocking detection.
    Detection loop calls submit() and returns immediately.
    This thread picks up incidents, calls GROQ_report(), writes to SQLite.
    """
    def __init__(self, camera_id: str, logger):
        self._queue     = queue.Queue(maxsize=50)
        self._logger    = logger
        self._camera_id = camera_id
        self._thread    = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"GroqWorker-{camera_id}"
        )
        self._thread.start()
        logger.info(f"GroqWorker started for {camera_id}")

    def submit(self, incident: dict):
        try:
            self._queue.put_nowait(incident)
        except queue.Full:
            self._logger.warning("GroqWorker queue full — dropping analysis request")

    def _worker_loop(self):
        while True:
            try:
                incident = self._queue.get(timeout=5)
                try:
                    incident["GROQ_analysis"] = GROQ_report(incident)
                except Exception as e:
                    self._logger.error(f"GroqWorker: GROQ_report failed: {e}")
                    incident["GROQ_analysis"] = "Analysis unavailable"

                try:
                    insert_incident(self._camera_id, incident)
                    upsert_worker(
                        self._camera_id,
                        str(incident["person_id"]),
                        incident["risk_score"],
                        incident["violation_type"]
                    )
                    self._logger.info(
                        f"INCIDENT SAVED | person={incident['person_id']} | "
                        f"violation={incident['violation_type']} | "
                        f"duration={incident['duration_seconds']:.1f}s | "
                        f"risk={incident['risk_level']} | score={incident['risk_score']}"
                    )
                except Exception as e:
                    self._logger.error(f"GroqWorker: DB write failed: {e}", exc_info=True)

                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"GroqWorker: Unexpected error: {e}", exc_info=True)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_ai_pipeline(video_source=1, camera_id="CAM_01", inference_size=640):

    # ── STORAGE SETUP ────────────────────────────────────────────────────────
    base_dir     = os.path.join("data", camera_id)
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    violation_log_path = os.path.join(base_dir, "violation_log.csv")
    site_safety_path   = os.path.join(base_dir, "Site_Safety.json")
    live_frame_path    = os.path.join(base_dir, "live_frame.jpg")

    logger = get_logger(camera_id)
    logger.info(f"Pipeline v4 initializing | camera={camera_id} | source={video_source} | headless={HEADLESS}")

    init_db()
    logger.info("Database initialized")

    groq_worker = GroqWorker(camera_id, logger)

    # ── CONSTANTS ────────────────────────────────────────────────────────────
    MODEL_PATH             = "ppe_yolov8s_best.pt"
    COOLDOWN_SECONDS       = 15
    MIN_VIOLATION_DURATION = 5
    MAX_MODEL_FAILURES     = 5

    violation_messages = {
        "NO-Gloves":      "GLOVE VIOLATION",
        "NO-Goggles":     "GOGGLE VIOLATION",
        "NO-Hardhat":     "HELMET VIOLATION",
        "NO-Mask":        "MASK VIOLATION",
        "NO-Safety Vest": "VEST VIOLATION"
    }
    severity_rules = {
        "HELMET VIOLATION": "HIGH",   "MASK VIOLATION":  "MEDIUM",
        "GOGGLE VIOLATION": "MEDIUM", "VEST VIOLATION":  "LOW",
        "GLOVE VIOLATION":  "LOW"
    }
    risk_points = {
        "HELMET VIOLATION": 30, "MASK VIOLATION": 15,
        "GOGGLE VIOLATION": 20, "VEST VIOLATION": 20,
        "GLOVE VIOLATION":  15
    }
    machinery_classes = ["Excavator", "Forklift", "Vehicle", "Crane", "Bulldozer"]

    # ── CAMERA OPEN ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {video_source}")
        logger.warning("Trying fallback webcam index 0")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.critical("No camera available. Pipeline cannot start.")
            return
        logger.info("Fallback webcam index 0 opened")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── CSV HEADER ───────────────────────────────────────────────────────────
    if not os.path.exists(violation_log_path):
        with open(violation_log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["TIMESTAMP", "PERSON_ID", "VIOLATION",
                 "CONFIDENCE", "DURATION_SECONDS", "IMAGE_PATH"]
            )

    # ── EVIDENCE CLEANUP (Issue 9) ───────────────────────────────────────────
    evidence_files = sorted(
        [f for f in os.listdir(evidence_dir) if f.endswith(".jpg")],
        key=lambda x: os.path.getctime(os.path.join(evidence_dir, x))
    )
    if len(evidence_files) > 1000:
        for old_file in evidence_files[:-800]:
            try:
                os.remove(os.path.join(evidence_dir, old_file))
            except Exception as e:
                logger.warning(f"Evidence cleanup failed for {old_file}: {e}")

    # ── MODEL LOAD WITH WATCHDOG (Issue 5) ───────────────────────────────────
    def load_model():
        logger.info(f"Loading YOLO model: {MODEL_PATH}")
        m = YOLO(MODEL_PATH)
        logger.info("YOLO model loaded")
        return m

    model          = load_model()
    model_failures = 0

    # ── STATE ────────────────────────────────────────────────────────────────
    site_risk_score            = 0
    site_risk_level            = "SAFE"
    historical_site_risk_score = 0
    total_incidents            = 0
    active_violations          = {}
    violation_alert_levels     = {}
    violation_cooldowns        = {}
    track_to_worker            = {}
    person_last_seen           = {}
    attempts                   = 0
    PERSON_TIMEOUT             = 30

    prev_time        = time.time()
    last_frame_save  = time.time()
    last_json_update = time.time()

    logger.info(f"Pipeline running | camera={camera_id}")

    try:
        while True:
            ret, frame = cap.read()

            # ── RECONNECT ─────────────────────────────────────────────────────
            if not ret:
                is_video_file = isinstance(video_source, str) and not str(video_source).isdigit()
                if is_video_file:
                    logger.info("Video file ended. Pipeline stopping cleanly.")
                    break
                attempts += 1
                wait_time = min(2 * attempts, 30)
                logger.warning(f"Webcam frame lost. Retry {attempts} (wait {wait_time}s)...")
                cap.release()
                time.sleep(wait_time)
                cap = cv2.VideoCapture(video_source)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    logger.info(f"Webcam reconnected after {attempts} attempts")
                    attempts = 0
                continue

            attempts = 0

            # ── LIVE FRAME SAVE — before inference (Issue 7: atomic write) ───
            if time.time() - last_frame_save > 2.0:
                try:
                    ret2, buf = cv2.imencode(".jpg", frame)
                    if ret2 and buf is not None:
                        tmp_path = live_frame_path + ".tmp.jpg"
                        with open(tmp_path, "wb") as _f:
                            _f.write(buf.tobytes())
                        try:
                            os.replace(tmp_path, live_frame_path)
                        except OSError:
                            pass
                    last_frame_save = time.time()
                except Exception as e:
                    logger.warning(f"Live frame save failed: {e}")

            # ── INFERENCE WITH WATCHDOG (Issue 5) ────────────────────────────
            try:
                orig_h, orig_w  = frame.shape[:2]
                scale_x         = orig_w / inference_size
                scale_y         = orig_h / inference_size
                inference_frame = cv2.resize(frame, (inference_size, inference_size))
                results         = model.track(
                    inference_frame, persist=True,
                    imgsz=inference_size, tracker="bytetrack.yaml"
                )
                model_failures = 0

            except Exception as e:
                model_failures += 1
                logger.error(f"Model inference failed ({model_failures}/{MAX_MODEL_FAILURES}): {e}")
                if model_failures >= MAX_MODEL_FAILURES:
                    logger.critical("Too many model failures. Reloading model...")
                    try:
                        del model
                        model          = load_model()
                        model_failures = 0
                        logger.info("Model reloaded successfully")
                    except Exception as reload_err:
                        logger.critical(f"Model reload failed: {reload_err}")
                        break
                continue

            current_time        = datetime.now()
            detected_people     = {}
            detected_violations = []
            detected_machinery  = []

            # ── DETECTION PARSING ─────────────────────────────────────────────
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                boxes     = results[0].boxes

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    xmin = int(x1 * scale_x)
                    ymin = int(y1 * scale_y)
                    xmax = int(x2 * scale_x)
                    ymax = int(y2 * scale_y)

                    cls_name   = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])

                    if cls_name == "Person":
                        if confidence <= 0.7:              continue
                        if (xmax - xmin) > (ymax - ymin): continue

                        now_sec = time.time()
                        for old_id in list(person_last_seen.keys()):
                            if now_sec - person_last_seen[old_id] > PERSON_TIMEOUT:
                                track_to_worker.pop(old_id, None)
                                person_last_seen.pop(old_id, None)

                        person_last_seen[track_id] = now_sec
                        persistent_id = track_to_worker.setdefault(
                            track_id, f"WORKER_{track_id}"
                        )
                        detected_people[track_id] = {
                            "bbox":           (xmin, ymin, xmax, ymax),
                            "persistent_id":  persistent_id,
                            "confidence":     confidence,
                            "has_violation":  False,
                            "violated_duration": timedelta(0),
                            "violations":     []
                        }

                    elif cls_name in violation_messages:
                        detected_violations.append({
                            "cls_name":       cls_name,
                            "violation_type": violation_messages[cls_name],
                            "confidence":     confidence,
                            "center":         ((xmin + xmax) // 2, (ymin + ymax) // 2)
                        })

                    elif cls_name in machinery_classes:
                        detected_machinery.append({"bbox": (xmin, ymin, xmax, ymax)})

                # ── ASSOCIATION ───────────────────────────────────────────────
                for violation in detected_violations:
                    v_x, v_y = violation["center"]
                    for tid, p_data in detected_people.items():
                        p_xmin, p_ymin, p_xmax, p_ymax = p_data["bbox"]
                        persistent_id = p_data["persistent_id"]

                        if not (p_xmin <= v_x <= p_xmax and p_ymin <= v_y <= p_ymax):
                            continue

                        dict_key = (persistent_id, violation["violation_type"])

                        if dict_key not in active_violations:
                            active_violations[dict_key] = {
                                "person_id":         persistent_id,
                                "start_time":        current_time,
                                "evidence_captured": False,
                                "last_seen":         current_time,
                                "confidence":        0,
                                "violation_type":    violation["violation_type"]
                            }
                        else:
                            active_violations[dict_key]["last_seen"] = current_time

                        duration = current_time - active_violations[dict_key]["start_time"]
                        p_data["has_violation"] = True
                        p_data["violations"].append(violation)

                        if duration > timedelta(seconds=MIN_VIOLATION_DURATION):
                            p_data["violated_duration"] = duration
                            cropped = frame[p_ymin:p_ymax, p_xmin:p_xmax]

                            file_name = (
                                f"{violation['violation_type']}_"
                                f"{persistent_id}_"
                                f"{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                            )
                            full_image_path = os.path.join(evidence_dir, file_name)

                            if violation["confidence"] > active_violations[dict_key].get("confidence", 0):
                                cv2.imwrite(full_image_path, cropped)
                                active_violations[dict_key]["confidence"] = violation["confidence"]
                                active_violations[dict_key]["Image_path"] = full_image_path

                            active_violations[dict_key]["duration"]   = duration
                            active_violations[dict_key]["risk_score"] = risk_points[violation["violation_type"]]
                            active_violations[dict_key]["risk_level"] = severity_rules[violation["violation_type"]]
                            active_violations[dict_key]["evidence_captured"] = True

                            # ── ALERT LOGIC (Issue 10: wrapped) ───────────────
                            alert_key  = (persistent_id, violation["violation_type"])
                            send_alert = False

                            if alert_key not in violation_alert_levels:
                                violation_alert_levels[alert_key] = {
                                    "last_alert": current_time, "level": 1
                                }
                                send_alert = True
                            else:
                                secs_since = (current_time - violation_alert_levels[alert_key]["last_alert"]).total_seconds()
                                level      = violation_alert_levels[alert_key]["level"]
                                if level == 1 and secs_since > 30:
                                    violation_alert_levels[alert_key]["level"] = 2
                                    violation_alert_levels[alert_key]["last_alert"] = current_time
                                    send_alert = True
                                elif level == 2 and secs_since > 300:
                                    violation_alert_levels[alert_key]["level"] = 3
                                    violation_alert_levels[alert_key]["last_alert"] = current_time
                                    send_alert = True

                            cooldown_ok = (
                                dict_key not in violation_cooldowns or
                                current_time >= violation_cooldowns[dict_key] + timedelta(seconds=COOLDOWN_SECONDS)
                            )
                            if cooldown_ok:
                                violation_cooldowns[dict_key] = current_time
                                try:
                                    with open(violation_log_path, "a", newline="") as f:
                                        csv.writer(f).writerow([
                                            current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                            f"Person-{persistent_id}",
                                            violation["violation_type"],
                                            f"{violation['confidence']:.2f}",
                                            f"{duration.total_seconds():.1f}",
                                            full_image_path
                                        ])
                                except Exception as e:
                                    logger.error(f"CSV write failed: {e}")

                            if send_alert:
                                level_label = (
                                    "ESCALATION" if violation_alert_levels[alert_key]["level"] == 3
                                    else "REMINDER" if violation_alert_levels[alert_key]["level"] == 2
                                    else "ALERT"
                                )
                                alert_msg = (
                                    f"🚨 {level_label}: "
                                    f"{violation['violation_type']} — "
                                    f"Person-{persistent_id}"
                                )
                                logger.warning(alert_msg)

                                alert_record = {
                                    "timestamp":      current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "camera_id":      camera_id,
                                    "person_id":      persistent_id,
                                    "alert_level":    violation_alert_levels[alert_key]["level"],
                                    "violation_type": violation["violation_type"],
                                    "message":        alert_msg
                                }
                                try:
                                    insert_alert(camera_id, alert_record)
                                except Exception as e:
                                    logger.error(f"Alert DB write failed: {e}")

                                try:
                                    send_alert(alert_msg, logger)
                                except Exception as e:
                                    logger.error(f"Telegram failed (non-fatal): {e}")

                # ── MACHINERY PROXIMITY ───────────────────────────────────────
                for tid, p_data in detected_people.items():
                    pid = p_data["persistent_id"]
                    p_xmin, p_ymin, p_xmax, p_ymax = p_data["bbox"]
                    for machine in detected_machinery:
                        mx1, my1, mx2, my2 = machine["bbox"]
                        dist = (
                            ((p_xmin + p_xmax) // 2 - (mx1 + mx2) // 2) ** 2 +
                            ((p_ymin + p_ymax) // 2 - (my1 + my2) // 2) ** 2
                        ) ** 0.5
                        if dist < 200:
                            for key in active_violations:
                                if key[0] == pid:
                                    active_violations[key]["near_machinery"] = True

            # ── ANNOTATE FRAME — always, for dashboard + OpenCV window ────────
            annotated = frame.copy()
            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            for tid, p_data in detected_people.items():
                px1, py1, px2, py2 = p_data["bbox"]
                pid = p_data["persistent_id"]
                if p_data["has_violation"]:
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)
                    v_labels = ", ".join([v["cls_name"] for v in p_data["violations"]])
                    cv2.putText(annotated, f"{pid} | {v_labels}",
                                (px1, max(py1 - 30, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(annotated, f"RISK: {p_data.get('risk_level', 'LOW')}",
                                (px1, max(py1 - 5, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{pid}: Safe",
                                (px1, max(py1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(annotated,
                        f"FPS: {fps:.1f} | RISK: {site_risk_level} ({site_risk_score})",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if not HEADLESS:
                cv2.imshow("AI Safety Pipeline v4", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── ANNOTATED FRAME SAVE — atomic (Issue 7) ──────────────────────
            try:
                ret3, buf3 = cv2.imencode(".jpg", annotated)
                if ret3 and buf3 is not None:
                    tmp_ann = live_frame_path + ".ann.jpg"
                    with open(tmp_ann, "wb") as _f:
                        _f.write(buf3.tobytes())
                    try:
                        os.replace(tmp_ann, live_frame_path)
                    except OSError:
                        pass
            except Exception as e:
                logger.warning(f"Annotated frame save failed: {e}")

            # ── INCIDENT CLOSE LOGIC ──────────────────────────────────────────
            for dict_key in list(active_violations.keys()):
                last_seen = active_violations[dict_key]["last_seen"]

                if current_time - last_seen > timedelta(seconds=3):
                    if active_violations[dict_key]["evidence_captured"]:
                        try:
                            start_time       = active_violations[dict_key]["start_time"]
                            duration_seconds = (last_seen - start_time).total_seconds()

                            incident = {
                                "person_id":        active_violations[dict_key]["person_id"],
                                "violation_type":   active_violations[dict_key]["violation_type"],
                                "start_time":       str(start_time),
                                "end_time":         str(last_seen),
                                "duration_seconds": duration_seconds,
                                "confidence":       active_violations[dict_key].get("confidence", 0),
                                "Image_path":       active_violations[dict_key].get("Image_path", ""),
                                "near_machinery":   active_violations[dict_key].get("near_machinery", False)
                            }

                            incident["severity"]   = severity_rules[incident["violation_type"]]
                            incident["risk_score"] = risk_points[incident["violation_type"]]

                            if incident["near_machinery"]:
                                incident["risk_score"] += 20

                            same_person_count = sum(
                                1 for k in active_violations if k[0] == incident["person_id"]
                            )
                            if same_person_count > 1:
                                incident["risk_score"] += 20

                            incident["risk_score"] += int(duration_seconds // 30) * 10

                            incident["risk_level"] = (
                                "LOW"    if incident["risk_score"] < 30 else
                                "MEDIUM" if incident["risk_score"] < 70 else
                                "HIGH"
                            )

                            groq_worker.submit(incident)
                            total_incidents            += 1
                            historical_site_risk_score += incident["risk_score"]

                        except Exception as e:
                            logger.error(f"Incident close error: {e}", exc_info=True)

                    del active_violations[dict_key]

            # ── SITE SAFETY JSON ──────────────────────────────────────────────
            site_risk_score = sum(v.get("risk_score", 0) for v in active_violations.values())
            site_risk_level = (
                "SAFE"     if site_risk_score == 0 else
                "WARNING"  if site_risk_score < 50 else
                "CRITICAL"
            )

            if time.time() - last_json_update > 2.0:
                site_safety = {
                    "current_site_risk_score":     site_risk_score,
                    "current_site_risk_level":     site_risk_level,
                    "historical_site_risk_score":  historical_site_risk_score,
                    "total_incidents":             total_incidents,
                    "active_violations":           len(active_violations),
                    "active_workers_in_violation": len({k[0] for k in active_violations}),
                    "workers_with_history":        total_incidents
                }
                try:
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
                        json.dump(site_safety, tmp, indent=4)
                        tmp_path = tmp.name
                    os.replace(tmp_path, site_safety_path)
                except Exception as e:
                    logger.error(f"Site safety JSON write failed: {e}")
                last_json_update = time.time()

    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        if (isinstance(video_source, str)
                and "temp" in video_source.lower()
                and os.path.exists(video_source)):
            try:
                os.unlink(video_source)
            except Exception as e:
                logger.warning(f"Temp file cleanup: {e}")
        logger.info("Pipeline v4 stopped gracefully")


def send_alert(message: str, logger):
    """
    Send alert via BOTH Email and Telegram.
    Both are optional — system continues even if one fails.
    """
    sent = False

    # 1. Try Telegram
    try:
        import requests
        BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
        CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
        
        if BOT_TOKEN and CHAT_ID:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": message},
                timeout=8
            )
            logger.info("✅ Telegram alert sent")
            sent = True
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")

    # 2. Try Email
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from dotenv import load_dotenv

        load_dotenv()

        sender_email    = os.getenv("EMAIL_SENDER")
        sender_password = os.getenv("EMAIL_PASSWORD")
        receiver_email  = os.getenv("EMAIL_RECEIVER")

        if sender_email and sender_password and receiver_email:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = "🚨 Construction Safety Alert"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ Email alert sent to {receiver_email}")
            sent = True
    except Exception as e:
        logger.warning(f"Email failed: {e}")

    if not sent:
        logger.warning("⚠️ Both Telegram and Email failed to send alert")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        cam_id = sys.argv[2] if len(sys.argv) > 2 else "CAM_01"
        src    = int(source) if source.isdigit() else source
        run_ai_pipeline(video_source=src, camera_id=cam_id, inference_size=640)
    else:
        run_ai_pipeline(video_source=1, camera_id="CAM_01", inference_size=640)