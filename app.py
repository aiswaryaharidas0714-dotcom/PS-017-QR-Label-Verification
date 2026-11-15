
import streamlit as st
import cv2
import numpy as np
import pytesseract
from database import connect_database

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
st.set_page_config(page_title="PS-017 Label Verification", layout="centered")

def verify_appearance(uploaded_image):
    reference_path = "captured_images/sample_product_label.png"
    reference_image = cv2.imread(reference_path)

    if reference_image is None:
        return "REVIEW", "Reference label not found."

    if uploaded_image.shape != reference_image.shape:
        return "REVIEW", "Image dimensions do not match."

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(reference_gray, uploaded_gray)
    _, threshold_image = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    changed_pixels = cv2.countNonZero(threshold_image)
    total_pixels = threshold_image.size
    difference_percentage = (changed_pixels / total_pixels) * 100

    if difference_percentage <= 0.5:
        return "PASS", f"Visual difference: {difference_percentage:.2f}%"

    return "FAIL", f"Visual difference: {difference_percentage:.2f}%"


st.title("Automated QR Label Verification")
st.write("Upload a product label to inspect it.")

uploaded_file = st.file_uploader(
    "Choose a product label image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read the uploaded image.")

    else:
        st.image(image, caption="Uploaded Product Label", use_container_width=True)

        if st.button("Inspect Product"):
            detector = cv2.QRCodeDetector()
            qr_data, points, _ = detector.detectAndDecode(image)

            if not qr_data:
                st.error("QR Code Not Detected")
                st.stop()

            st.success("QR Code Detected")
            st.write("Product ID:", qr_data)

            connection = connect_database()

            if connection is None:
                st.error("Database connection failed.")
                st.stop()

            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM products WHERE product_id = %s",
                (qr_data,)
            )
            product = cursor.fetchone()

            if product is None:
                st.error("Product not found in database.")
                cursor.close()
                connection.close()
                st.stop()

            st.subheader("Product Details")
            st.write("Product ID:", product[0])
            st.write("Product Name:", product[1])
            st.write("Batch Number:", product[2])
            st.write("Manufacturing Date:", product[3])
            st.write("Expiry Date:", product[4])

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enlarged = cv2.resize(
                gray, None, fx=2, fy=2,
                interpolation=cv2.INTER_CUBIC
            )
            ocr_text = pytesseract.image_to_string(enlarged)

            st.subheader("OCR Extracted Text")
            st.text(ocr_text if ocr_text.strip() else "No text extracted.")

            expected_values = [
                str(product[1]),
                str(product[2])
            ]

            normalized_ocr = ocr_text.upper().replace(" ", "").replace("\n", "")

            text_status = (
                "PASS"
                if all(value.upper().replace(" ", "") in normalized_ocr
                       for value in expected_values)
                else "REVIEW"
            )

            qr_status = "PASS"
            appearance_status, appearance_remarks = verify_appearance(image)

            if (
                text_status == "FAIL"
                or appearance_status == "FAIL"
            ):
                final_result = "FAIL"

            elif (
                text_status == "REVIEW"
                or appearance_status == "REVIEW"
            ):
                final_result = "REVIEW"

            else:
                final_result = "PASS"

            remarks = appearance_remarks

            cursor.execute(
                """INSERT INTO inspection_history
                (product_id, qr_status, text_status,
                 appearance_status, final_result, remarks)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    qr_data, qr_status, text_status,
                    appearance_status, final_result, remarks
                )
            )

            connection.commit()
            cursor.close()
            connection.close()

            st.subheader("Inspection Results")
            st.write("QR Verification:", qr_status)
            st.write("Text Verification:", text_status)
            st.write("Appearance Verification:", appearance_status)
            st.write("Appearance Remarks:", appearance_remarks)

            if final_result == "PASS":
                st.success("Final Result: PASS")
            elif final_result == "FAIL":
                st.error("Final Result: FAIL")
            else:
                st.warning("Final Result: REVIEW")

            st.info("Inspection result saved in database.")


st.divider()
st.subheader("Inspection History")

history_connection = connect_database()

if history_connection:
    history_cursor = history_connection.cursor()

    history_cursor.execute("""
        SELECT inspection_id, product_id, inspection_date,
               qr_status, text_status, appearance_status, final_result
        FROM inspection_history
        ORDER BY inspection_id DESC
    """)

    history = history_cursor.fetchall()

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
        column_config={
            "inspection_id": "Inspection ID",
            "product_id": "Product ID",
            "inspection_date": "Inspection Date",
            "qr_status": "QR Status",
            "text_status": "Text Status",
            "appearance_status": "Appearance Status",
            "final_result": "Final Result"
        }
    )

    history_cursor.close()
    history_connection.close()

else:
    st.error("Unable to load inspection history.")
