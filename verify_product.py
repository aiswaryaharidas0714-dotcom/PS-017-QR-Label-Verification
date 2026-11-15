import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image = cv2.imread("captured_images/label_20261115_161652.jpg")

detector = cv2.QRCodeDetector()

data, points, straight_qrcode = detector.detectAndDecode(image)

if data:
    print("QR Code Detected!")
    print("Product ID:", data)
else:
    print("QR Code Not Detected!")
from database import connect_database

if data:
    connection = connect_database()
    cursor = connection.cursor()

    query = "SELECT * FROM products WHERE product_id = %s"
    cursor.execute(query, (data,))

    product = cursor.fetchone()

    if product:
        print("\nProduct Details:")
        print("Product ID:", product[0])
        print("Product Name:", product[1])
        print("Batch Number:", product[2])
        print("Manufacturing Date:", product[3])
        print("Expiry Date:", product[4])
    else:
        print("Product Not Found in Database!")

    cursor.close()
    connection.close()   