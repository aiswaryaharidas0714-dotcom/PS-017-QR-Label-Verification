import cv2

image = cv2.imread("captured_images/label_20261115_161652.jpg")
if image is None:
    print("Label image not found!")
    exit()

detector = cv2.QRCodeDetector()

data, points, _ = detector.detectAndDecode(image)

if data:
    print("QR Code Detected!")
    print("QR Data:", data)
else:
    print("QR Code Not Detected!")