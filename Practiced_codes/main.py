from ultralytics import YOLO
import cv2

# yolov8n.pt (Nano): Fastest, lowest resource usage, slightly less accurate.
# yolov8s.pt (Small): Better accuracy, still very fast.
# yolov8m.pt (Medium): Balanced performance.
# yolov8l.pt (Large): High accuracy, requires more processing power.
# yolov8x.pt (Extra Large): Highest accuracy, slowest, requires powerful GPUs.



model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    persons=[]
    no_persons=0
    ret, frame = cap.read()#ret will tell TRUE OR FALSE if the frame is read correctly, frame will contain the actual image data

    if not ret:
        break

    results = model(frame)
    for box in results[0].boxes:#results[0] will give us the first image in the batch, .boxes will give us the bounding boxes for the detected objects
        confidence = float(box.conf[0])#box.conf will give us the confidence score of the detected object, we convert it to float because it is in tensor format
        if confidence <=0.5:
            continue
        cls_id = int(box.cls[0])#box.cls will give us the class ID of the detected object, we convert it to int because it is in tensor format
        class_name=model.names[cls_id]
        if class_name !="person":
            continue
        detection={"class name":class_name,"confidence":confidence,"box":(box.xyxy[0])}
        persons.append(detection)
        #print(cls_id)
        #print(model.names[cls_id])#model.names will give us the class name corresponding to the class ID
        xmin,ymin,xmax,ymax = map(int, box.xyxy[0])#box.xyxy will give us the coordinates of the bounding box in the format (x1, y1, x2, y2), xmin,ymin,xmax,ymax
        
        cv2.rectangle(
            frame,
            (xmin, ymin),
            (xmax, ymax),
            (0, 0, 255),#, 0),
            2
        )
        label=f"{model.names[cls_id]}: {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            f"NO_OF _PERSONS:{len(persons)}     "
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        print(model.names[cls_id], confidence, (xmin, ymin, xmax, ymax))
    #annotated_frame = results[0].plot()

    cv2.imshow("Custom Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 

from collections import Counter
import os

counter = Counter()

labels_dir = r"C:\Users\Anuj\Documents\train\labels"

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        with open(os.path.join(labels_dir, file), "r") as f:
            for line in f:
                cls = int(line.split()[0])
                counter[cls] += 1

print(counter)