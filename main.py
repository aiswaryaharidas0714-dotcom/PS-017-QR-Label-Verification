
import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
import shutil
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
from database import connect_database

# Find Tesseract on the hosting system
tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists():
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

st.set_page_config(
    page_title="PS-017 Label Verification",
    layout="centered"
)

MAX_PIXELS = 12_000_000
MAX_DIMENSION = 1600

def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def load_uploaded_image(uploaded_file):
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue()))

        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image resolution is too large.")

        image = ImageOps.exif_transpose(image)
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        image = image.convert("RGB")

        image_array = np.array(image)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    except Exception as e:
        st.error(f"Unable to process image: {e}")
        return None


def verify_appearance(uploaded_image):
    reference_path = "captured_images/sample_product_label.png"
    reference_image = cv2.imread(reference_path)

    if reference_image is None:
        return "REVIEW", "Reference label not found."

    if uploaded_image.shape != reference_image.shape:
        return "REVIEW", "Image dimensions do not match."

    reference_gray = cv2.cvtColor(
        reference_image, cv2.COLOR_BGR2GRAY
    )
    uploaded_gray = cv2.cvtColor(
        uploaded_image, cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(reference_gray, uploaded_gray)

    _, threshold_image = cv2.threshold(
        difference, 30, 255, cv2.THRESH_BINARY
    )

    changed_pixels = cv2.countNonZero(threshold_image)
    total_pixels = threshold_image.size

    difference_percentage = (
        changed_pixels / total_pixels
    ) * 100

    if difference_percentage <= 0.5:
        return "PASS", f"Visual difference: {difference_percentage:.2f}%"

    return "FAIL", f"Visual difference: {difference_percentage:.2f}%"


def save_inspection(product_id, qr_status, text_status,
                    appearance_status, final_result, remarks):

    connection = connect_database()

    if connection is None:
        raise RuntimeError("Database connection failed.")

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            """INSERT INTO inspection_history
            (product_id, qr_status, text_status,
             appearance_status, final_result, remarks)
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                product_id, qr_status, text_status,
                appearance_status, final_result, remarks
            )
        )

        connection.commit()

    finally:
        if cursor:
            cursor.close()
        connection.close()


st.title("Automated QR Label Verification")
st.write("Upload a product label to inspect it.")

uploaded_file = st.file_uploader(
    "Choose a product label image",
    type=["jpg", "jpeg", "png"],
    max_upload_size=5
)

if uploaded_file is not None:

    image = load_uploaded_image(uploaded_file)

    if image is not None:

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Product Label",
            use_container_width=True
        )

        if st.button("Inspect Product"):

            with st.spinner("Inspecting product label..."):

                detector = cv2.QRCodeDetector()
                qr_data, points, _ = detector.detectAndDecode(image)

                if not qr_data:
                    st.error("QR Code Not Detected")
                    st.stop()

                qr_data = qr_data.strip()
                st.success("QR Code Detected")
                st.write("Product ID:", qr_data)

                connection = connect_database()

                if connection is None:
                    st.error("Database connection failed.")
                    st.stop()

                cursor = None

                try:
                    cursor = connection.cursor()

                    cursor.execute(
                        "SELECT * FROM products WHERE product_id = %s",
                        (qr_data,)
                    )

                    product = cursor.fetchone()

                except Exception as e:
                    st.error(f"Database error: {e}")
                    st.stop()

                finally:
                    if cursor:
                        cursor.close()
                    connection.close()

                if product is None:
                    st.error("Product not found in database.")
                    st.stop()

                st.subheader("Product Details")
                st.write("Product ID:", product[0])
                st.write("Product Name:", product[1])
                st.write("Batch Number:", product[2])
                st.write("Manufacturing Date:", product[3])
                st.write("Expiry Date:", product[4])

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                try:
                    ocr_text = pytesseract.image_to_string(
                        gray,
                        config="--oem 3 --psm 6",
                        timeout=15
                    )
                except Exception as e:
                    st.error(f"OCR processing failed: {e}")
                    st.stop()

                st.subheader("OCR Extracted Text")
                st.text(
                    ocr_text if ocr_text.strip()
                    else "No text extracted."
                )

                expected_values = [
                    str(product[1]),
                    str(product[2])
                ]

                normalized_ocr = normalize(ocr_text)

                text_status = (
                    "PASS"
                    if all(
                        normalize(value) in normalized_ocr
                        for value in expected_values
                    )
                    else "REVIEW"
                )

                qr_status = "PASS"

                appearance_status, appearance_remarks = (
                    verify_appearance(image)
                )

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

                try:
                    save_inspection(
                        qr_data,
                        qr_status,
                        text_status,
                        appearance_status,
                        final_result,
                        appearance_remarks
                    )

                    st.info("Inspection result saved in database.")

                except Exception as e:
                    st.error(f"Unable to save inspection: {e}")

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


st.divider()
st.subheader("Inspection History")

history_connection = None
history_cursor = None

try:
    history_connection = connect_database()

    if history_connection is None:
        raise RuntimeError("Database connection failed.")

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

except Exception as e:
    st.error(f"Unable to load inspection history: {e}")

finally:
    if history_cursor:
        history_cursor.close()
    if history_connection:
        history_connection.close()
