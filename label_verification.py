import cv2
import pytesseract
import re
from database import connect_database

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image = cv2.imread("sample_product_label.png")

if image is None:
    print("Label image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

extracted_text = pytesseract.image_to_string(
    gray,
    config="--oem 3 --psm 6"
)

print("Extracted Label Text:")
print(extracted_text)
connection = connect_database()
cursor = connection.cursor()

query = "SELECT * FROM products WHERE product_id = %s"
cursor.execute(query, ("PROD001",))

product = cursor.fetchone()

if product:
    print("\nRegistered Product Details:")
    print("Product Name:", product[1])
    print("Batch Number:", product[2])
    print("Manufacturing Date:", product[3])
    print("Expiry Date:", product[4])
else:
    print("Product Not Found!")

cursor.close()
connection.close()
print("\nLabel Verification:")

clean_text = re.sub(r'[^A-Z0-9]', '', extracted_text.upper())

expected_details = {
    "Product Name": product[1],
    "Batch Number": product[2],
    "Manufacturing Date": product[3].strftime("%d-%m-%Y"),
    "Expiry Date": product[4].strftime("%d-%m-%Y")
}

all_valid = True

for field, expected in expected_details.items():
    expected = re.sub(r'[^A-Z0-9]', '', str(expected).upper())

    if expected in clean_text:
        print(field + ": MATCH")
    else:
        print(field + ": MISMATCH")
        all_valid = False

if all_valid:
    print("\nFINAL RESULT: PASS")
else:
    print("\nFINAL RESULT: FAIL")