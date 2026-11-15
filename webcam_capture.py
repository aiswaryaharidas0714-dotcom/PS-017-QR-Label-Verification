
import cv2
import os
from datetime import datetime

os.makedirs("captured_images", exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not detected!")
    exit()

print("Webcam started successfully!")
print("Press C to capture the label.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Unable to read webcam.")
        break

    cv2.imshow("PS-017 - Product Label Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = datetime.now().strftime("label_%Y%m%d_%H%M%S.jpg")
        filepath = os.path.join("captured_images", filename)

        cv2.imwrite(filepath, frame)
        print("Image saved:", filepath)

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()