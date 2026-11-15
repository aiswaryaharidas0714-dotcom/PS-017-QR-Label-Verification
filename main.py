
import cv2
import pytesseract
import re
from datetime import datetime
from pathlib import Path
from database import connect_database

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CAPTURE_FOLDER = Path("captured_images")

def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def capture_label():
    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Camera resolution:", width, "x", height)

    if not camera.isOpened():
        print("Webcam could not be opened!")
        return None

    print("Webcam started!")
    print("Press C to capture the label.")
    print("Press Q to quit.")

    captured_image = None

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Unable to read webcam!")
            break

        cv2.imshow("QR Label Inspection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            CAPTURE_FOLDER.mkdir(exist_ok=True)
            filename = "inspection_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            captured_image = str(CAPTURE_FOLDER / filename)
            cv2.imwrite(captured_image, frame)
            print("Image captured:", captured_image)
            break

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    return captured_image

def detect_qr(image):
    detector = cv2.QRCodeDetector()

    for scale in [1, 1.5, 2]:
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        methods = [
            gray,
            cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],
            cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 5
            )
        ]

        for processed in methods:
            data, points, _ = detector.detectAndDecode(processed)

            if data:
                return data, points

    return "", None

def inspect_label(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Captured image not found!")
        return

    qr_data, points = detect_qr(image)

    qr_status = "FAIL"
    text_status = "FAIL"
    appearance_status = "REVIEW"
    product_id = None
    remarks = []

    if qr_data:
        product_id = qr_data.strip()
        print("\nQR Data:", product_id)

        connection = connect_database()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM products WHERE product_id = %s",
            (product_id,)
        )

        product = cursor.fetchone()
        cursor.close()
        connection.close()

        if product:
            qr_status = "PASS"
            print("Product found in database!")

            resized = cv2.resize(
                image,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            gray = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            extracted_text = pytesseract.image_to_string(
                gray,
                config="--oem 3 --psm 3"
            )

            print("\nExtracted OCR Text:")
            print(extracted_text)

            checks = {
                "Product Name": product[1],
                "Batch Number": product[2],
                "Manufacturing Date": product[3].strftime("%d-%m-%Y"),
                "Expiry Date": product[4].strftime("%d-%m-%Y")
            }

            print("\nTEXT VERIFICATION:")

            clean_text = normalize(extracted_text)
            all_match = True

            for field, expected in checks.items():
                if normalize(expected) in clean_text:
                    print(field + ": MATCH")
                else:
                    print(field + ": MISMATCH")
                    all_match = False
                    remarks.append(field + " mismatch")

            text_status = "PASS" if all_match else "FAIL"

            appearance_status = "REVIEW"
            remarks.append("Appearance verification requires full label alignment")

        else:
            remarks.append("Product ID not registered in database")

    else:
        print("QR Code not detected!")
        remarks.append("QR code not detected")

    if qr_status == "FAIL":
        remarks.append("QR verification failed")

    if qr_status == "FAIL" or text_status == "FAIL":
        final_result = "FAIL"
    else:
        final_result = "REVIEW"

    connection = connect_database()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO inspection_history
        (product_id, qr_status, text_status, appearance_status,
         final_result, remarks)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        product_id,
        qr_status,
        text_status,
        appearance_status,
        final_result,
        "; ".join(remarks) if remarks else "All checks passed"
    ))

    connection.commit()

    print("\n========== INSPECTION RESULT ==========")
    print("QR Status:", qr_status)
    print("Text Status:", text_status)
    print("Appearance Status:", appearance_status)
    print("FINAL RESULT:", final_result)
    print("Remarks:", "; ".join(remarks) if remarks else "All checks passed")
    print("Inspection saved to database!")

    cursor.close()
    connection.close()

image_path = capture_label()

if image_path:
    inspect_label(image_path)
