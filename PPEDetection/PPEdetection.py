from ultralytics import YOLO
import cv2
import math

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("construction site.mp4")

model = YOLO("best (1).pt")
# model = YOLO("ppe.pt")
classNames = ['Glove', 'Helmet', 'No_BreathingApparatus', 'No_Glove', 'No_Goggles', 'No_Harness', 'No_Helmet', 'No_Shoe', 'Person', 'Safety_Harness', 'Shoe']
# classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if cls<12:
                currentClass = classNames[cls]
                print(currentClass)
                if conf>0.2:
                    myColor = (0, 0, 225)

                    putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                       colorT=(255,255,255),colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
