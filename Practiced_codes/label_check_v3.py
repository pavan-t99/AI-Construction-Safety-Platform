import cv2
from ultralytics import YOLO
import csv
import os
from datetime import datetime

if not os.path.exists("violation_log.csv"):
    with open("violation_log.csv", "a",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["TIMESTAMP","VIOLATION","CONFIDENCE"])

violation_messages={"NO-Gloves":"GLOVE VIOLATION",
                    "NO-Goggles":"GOGGLE VIOLATION",
                    "NO-Hardhat":"HELMET VIOLATION",
                    "NO-Mask":"MASK VIOLATION",
                    "NO-Safety Vest":"VEST VIOLATION"}

#image_path= r"C:\Users\Anuj\Documents\AI_Safety_System\train\images\-424-_png_jpg.rf.9f5c737706f9c6166fefe5d5aa5d292c.jpg"
model=YOLO("ppe_yolov8s_best.pt")
#image = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
#h, w, _ = image.shape
#result= model(image)
print(model.names)
violation_found = False
while True:
    ret,frame = cap.read()
    
    if not ret:
        print("FAILED to capture the frame")
        break

    results = model(frame)
    
    for box in results[0].boxes:
        confidence=float(box.conf[0])
        if confidence <=0.0:
            continue
        cls_id=int(box.cls[0])
        cls_name=model.names[cls_id]
        xmin,ymin,xmax,ymax=map(int,box.xyxy[0])    

        if cls_name in violation_messages:
            violation_found = True
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_name=f"{violation_messages[cls_name]}_{timestamp.replace(':','-')}.jpg"
            with open("violation_log.csv", "a", newline="") as f:
                writer=csv.writer(f)
                writer.writerow([timestamp, violation_messages[cls_name], confidence])
            cv2.rectangle(frame,
                    (xmin,ymin) ,
                    (xmax,ymax),
                    (0,0,255)
                    ,2)
            cv2.putText(frame,
                    f"{cls_name} : {confidence:.2f}",
                    (xmin,ymin-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,0,255),1)
            print(f"[{timestamp}] {violation_messages[cls_name]} with confidence {confidence:.2f} at coords {xmin,ymin,xmax,ymax}")
        else:
            cv2.rectangle(frame,
                    (xmin,ymin) ,
                    (xmax,ymax),
                    (0,255,0)
                    ,2)
            if cls_name.lower() != "person":
                print(f"{cls_name} satisfies safety requirements with confidence {confidence:.2f} at coords {xmin,ymin,xmax,ymax}")

            cv2.putText(frame,
                        f"{cls_name} : {confidence:.2f}",
                        (xmin,ymin-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0,255,0),1)
    
    

    if violation_found:
        cv2.imwrite(file_name, frame)
    cv2.imshow("Custom Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()