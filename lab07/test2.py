import cv2
cap = cv2.VideoCapture(1)
print("Camera opened:", cap.isOpened())
cap.release()