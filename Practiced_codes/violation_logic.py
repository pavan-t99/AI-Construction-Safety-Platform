import cv2
from datetime import datetime
import csv
import os
violation_messages = {
    6: "GLOVE VIOLATION",
    7: "GOGGLE VIOLATION",
    8: "HELMET VIOLATION",
    9: "MASK VIOLATION",
    10: "VEST VIOLATION"
}
names = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves',
         'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
         'Person', 'Safety Cone', 'Safety Vest']

#creating violation_log.csv file
if not os.path.exists("violation_log.csv"):
    with open("violation_log.csv", "a",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["TIMESTAMP","VIOLATION","CONFIDENCE"])
# IMAGE PATH
image_path = r"C:\Users\Anuj\Documents\train\image\image2.jpg"

# LABEL PATH
label_path = r"C:\Users\Anuj\Desktop\image2.txt"

image = cv2.imread(image_path)#image is a numpy array of shape (height, width, channels) where channels is usually 3 for RGB images.

h, w, _ = image.shape

with open(label_path, "r") as f:
    labels = f.readlines()
violation_found = False
for label in labels:

    data = label.strip().split()

    class_id = int(data[0])

    x_center = float(data[1])
    y_center = float(data[2])
    box_width = float(data[3])
    box_height = float(data[4])

    # YOLO normalized -> pixels

    x_center = x_center * w
    y_center = y_center * h

    box_width = box_width * w
    box_height = box_height * h

    xmin = int(x_center - box_width / 2)
    ymin = int(y_center - box_height / 2)

    xmax = int(x_center + box_width / 2)
    ymax = int(y_center + box_height / 2)

    cv2.rectangle(
        image,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        str(class_id),
        (xmin, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )
    
    if class_id in violation_messages:
        violation_found = True
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name=f"violation at {timestamp}.jpg"
        with open("violation_log.csv","a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([timestamp,violation_messages[class_id],90])#confidence])
        print(f"[{timestamp}] {violation_messages[class_id]}")
    else:
         class_name = names[class_id]
        print(f"{class_name} satisfies safety requirements ")

if violation_found:
    cv2.imwrite(
        file_name,
        image )
cv2.imshow("Annotations", image)

cv2.waitKey(0)
cv2.destroyAllWindows()