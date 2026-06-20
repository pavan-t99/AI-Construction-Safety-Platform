import cv2

# IMAGE PATH
image_path = r"C:\Users\Anuj\Documents\train\images\PP02img886_jpg.rf.e1070d72342c9d7b93e96b77ce548abd.jpg"

# LABEL PATH
label_path = r"C:\Users\Anuj\Documents\train\labels\PP02img886_jpg.rf.e1070d72342c9d7b93e96b77ce548abd.txt"

image = cv2.imread(image_path)#image is a numpy array of shape (height, width, channels) where channels is usually 3 for RGB images.

h, w, _ = image.shape

with open(label_path, "r") as f:
    labels = f.readlines()

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

cv2.imshow("Annotations", image)

cv2.waitKey(0)
cv2.destroyAllWindows()