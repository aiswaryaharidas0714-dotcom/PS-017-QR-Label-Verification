import os
import streamlit as st
import cv2
import numpy as np
import pytesseract
from database import connect_database


# Tesseract configuration
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


st.set_page_config(
    page_title="PS-017 Label Verification",
    page_icon="🔍",
    layout="centered"
)


# ---------------- LIGHTING QUALITY CHECK ----------------

def check_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    glare = float(np.mean(gray >= 250) * 100)

    if brightness < 55:
        return "POOR", brightness, "Image is too dark. Increase lighting."

    if brightness > 220 or glare > 35:
        return "TOO BRIGHT", brightness, "Excessive brightness or glare detected."

    if contrast < 20:
        return "LOW CONTRAST", brightness, "Image clarity is low."

    return "GOOD", brightness, "Lighting conditions are suitable."


# ---------------- OCR TEXT VERIFICATION ----------------

def verify_text(image, product):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    denoised = cv2.GaussianBlur(enlarged, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    ocr_text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6"
    )

    normalized_ocr = "".join(ocr_text.upper().split())

    expected_values = [
        str(product[1]),
        str(product[2])
    ]

    text_status = (
        "PASS"
        if all(
            "".join(value.upper().split()) in normalized_ocr
            for value in expected_values
        )
        else "REVIEW"
    )

    return ocr_text, text_status


# ---------------- APPEARANCE VERIFICATION ----------------

def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


def isolate_label(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.12:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx.reshape(4, 2)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    corners = order_points(candidates[0][1])

    top_left, top_right, bottom_right, bottom_left = corners

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)

    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))

    if width < 100 or height < 70:
        return None

    aspect_ratio = width / float(height)

    if aspect_ratio < 0.5 or aspect_ratio > 3.0:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(corners, destination)

    warped = cv2.warpPerspective(image, transform, (width, height))

    return warped


def verify_appearance(image):
    reference_path = "captured_images/sample_product_label.png"
    reference = cv2.imread(reference_path)

    if reference is None:
        return "REVIEW", "Reference label not found."

    # Detect and isolate the captured label
    captured_label = isolate_label(image)

    if captured_label is None:
        return (
            "REVIEW",
            "Label boundary could not be detected reliably. "
            "Keep the complete label visible and capture again."
        )

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    captured_gray = cv2.cvtColor(captured_label, cv2.COLOR_BGR2GRAY)

    # Resize captured label to reference dimensions
    height, width = reference_gray.shape

    captured_gray = cv2.resize(
        captured_gray,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # Normalize lighting differences
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    reference_normalized = clahe.apply(reference_gray)
    captured_normalized = clahe.apply(captured_gray)

    # ORB feature detection
    orb = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=10
    )

    keypoints_ref, descriptors_ref = orb.detectAndCompute(
        reference_normalized,
        None
    )

    keypoints_cap, descriptors_cap = orb.detectAndCompute(
        captured_normalized,
        None
    )

    if descriptors_ref is None or descriptors_cap is None:
        return (
            "REVIEW",
            "Not enough visual features were detected for a reliable comparison."
        )

    # Match visual features
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    pairs = matcher.knnMatch(
        descriptors_cap,
        descriptors_ref,
        k=2
    )

    good_matches = []

    for pair in pairs:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    if len(good_matches) < 5:
        return (
            "REVIEW",
            f"Only {len(good_matches)} matching visual features found. "
            "Manual inspection recommended."
        )

    captured_points = np.float32(
        [keypoints_cap[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    reference_points = np.float32(
        [keypoints_ref[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    _, inlier_mask = cv2.findHomography(
        captured_points,
        reference_points,
        cv2.RANSAC,
        5.0
    )

    if inlier_mask is None:
        return (
            "REVIEW",
            "The label could not be aligned confidently. "
            "Manual inspection recommended."
        )

    inlier_count = int(inlier_mask.sum())
    match_ratio = inlier_count / len(good_matches)

    remarks = (
        f"Matching features: {len(good_matches)}; "
        f"aligned features: {inlier_count}; "
        f"alignment score: {match_ratio * 100:.1f}%."
    )

    if inlier_count >= 12 and match_ratio >= 0.30:
        return "PASS", remarks + " Label layout matches the reference."

    return "REVIEW", remarks + " Manual inspection recommended."


# ---------------- PRODUCT INSPECTION ----------------

def inspect_product(image):
    detector = cv2.QRCodeDetector()

    qr_data, points, _ = detector.detectAndDecode(image)

    if not qr_data:
        st.error("QR Code Not Detected.")
        return

    st.success("QR Code Detected")
    st.write("Product ID:", qr_data)

    connection = None
    cursor = None

    try:
        connection = connect_database()

        if connection is None:
            st.error("Unable to connect to the product database.")
            return

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT product_id, product_name, batch_number,
                   manufacturing_date, expiry_date
            FROM products
            WHERE product_id = %s
            """,
            (qr_data,)
        )

        product = cursor.fetchone()

        if product is None:
            st.error("Product not found in database.")
            return

        # Product details
        st.subheader("3. Product Database Validation")

        st.write("Product ID:", product[0])
        st.write("Product Name:", product[1])
        st.write("Batch Number:", product[2])
        st.write("Manufacturing Date:", product[3])
        st.write("Expiry Date:", product[4])

        st.success("Product Database Validation: PASS")

        # OCR
        st.subheader("4. OCR Text Verification")

        ocr_text, text_status = verify_text(image, product)

        st.text(
            ocr_text if ocr_text.strip()
            else "No text extracted."
        )

        st.write("Text Verification:", text_status)

        # Appearance
        st.subheader("5. Appearance Verification")

        appearance_status, appearance_remarks = verify_appearance(image)

        st.write(appearance_remarks)
        st.write("Appearance Verification:", appearance_status)

        # Final decision
        qr_status = "PASS"

        if text_status == "FAIL" or appearance_status == "FAIL":
            final_result = "FAIL"

        elif text_status == "REVIEW" or appearance_status == "REVIEW":
            final_result = "REVIEW"

        else:
            final_result = "PASS"

        # Save inspection history
        cursor.execute(
            """
            INSERT INTO inspection_history
            (product_id, qr_status, text_status,
             appearance_status, final_result, remarks)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                qr_data,
                qr_status,
                text_status,
                appearance_status,
                final_result,
                appearance_remarks
            )
        )

        connection.commit()

        # Final result
        st.subheader("6. Final Inspection Result")

        st.write("QR Verification:", qr_status)
        st.write("Text Verification:", text_status)
        st.write("Appearance Verification:", appearance_status)

        if final_result == "PASS":
            st.success("FINAL RESULT: PASS")

        elif final_result == "FAIL":
            st.error("FINAL RESULT: FAIL")

        else:
            st.warning("FINAL RESULT: REVIEW")

        st.info("Inspection successfully saved in MySQL.")

    except Exception as error:
        st.error("Inspection Error: " + str(error))

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ---------------- MAIN WEBSITE ----------------

st.title("🔍 Automated QR Label Verification")

st.write(
    "Webcam-based product label inspection using QR detection, "
    "database validation, OCR and appearance verification."
)

st.divider()

st.subheader("1. Capture Product Label")

st.write("Choose how you want to provide the product label.")

input_mode = st.radio(
    "Select input method",
    ["Webcam", "Upload Sample Image"],
    horizontal=True
)

image_file = None

if input_mode == "Webcam":

    st.write("Position the product label clearly in front of your webcam.")

    image_file = st.camera_input("Capture Product Label")

else:

    image_file = st.file_uploader(
        "Choose a product label image",
        type=["jpg", "jpeg", "png"]
    )


if image_file is not None:

    file_bytes = np.asarray(
        bytearray(image_file.getvalue()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("Unable to read the selected image.")

    else:

        # Resize large images
        height, width = image.shape[:2]

        if max(height, width) > 1200:

            scale = 1200 / max(height, width)

            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Captured Product Label",
            use_container_width=True
        )

        # Lighting
        st.subheader("2. Lighting Quality Check")

        lighting_status, brightness, lighting_message = check_lighting(image)

        st.write(f"Brightness Score: {brightness:.2f}")

        if lighting_status == "GOOD":
            st.success("Lighting Quality: GOOD")

        else:
            st.warning("Lighting Quality: " + lighting_status)

        st.write(lighting_message)

        # Inspection button
        if st.button(
            "Inspect Product",
            type="primary",
            use_container_width=True
        ):

            if input_mode == "Webcam" and lighting_status != "GOOD":

                st.warning(
                    "Please improve the lighting and capture the label again."
                )

            else:

                if input_mode == "Webcam":
                    st.success("Lighting check passed.")

                else:
                    st.info(
                        "Sample image mode: Lighting check is informational."
                    )

                inspect_product(image)


# ---------------- INSPECTION HISTORY ----------------

st.divider()

st.subheader("Inspection History")

history_connection = None
history_cursor = None

try:

    history_connection = connect_database()

    if history_connection is not None:

        history_cursor = history_connection.cursor()

        history_cursor.execute(
            """
            SELECT inspection_id, product_id, inspection_date,
                   qr_status, text_status, appearance_status, final_result
            FROM inspection_history
            ORDER BY inspection_id DESC
            """
        )

        history = history_cursor.fetchall()

        history_columns = [
            "Inspection ID",
            "Product ID",
            "Inspection Date",
            "QR Status",
            "Text Status",
            "Appearance Status",
            "Final Result"
        ]

        history_rows = [
            dict(zip(history_columns, row))
            for row in history
        ]

        if history_rows:

            st.dataframe(
                history_rows,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No inspection records found yet.")

    else:

        st.warning("Unable to load inspection history.")

except Exception as error:

    st.warning("History Error: " + str(error))

finally:

    if history_cursor is not None:
        history_cursor.close()

    if history_connection is not None:
        history_connection.close()