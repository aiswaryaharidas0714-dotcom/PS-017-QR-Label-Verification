import os
import re
import shutil
from datetime import datetime, date
from pathlib import Path
from difflib import SequenceMatcher
import cv2
import numpy as np
import pytesseract
from PIL import Image

from database import get_product

# ---------------- TESSERACT CONFIGURATION ----------------
def configure_tesseract():
    """Auto-detects Tesseract binary on Windows, Linux and macOS."""
    if os.name == "nt":
        standard_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for p in standard_windows_paths:
            if Path(p).exists():
                pytesseract.pytesseract.tesseract_cmd = p
                return p
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return which_path
    if Path("/usr/bin/tesseract").exists():
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
        return "/usr/bin/tesseract"
    return None

TESSERACT_EXE = configure_tesseract()


# ---------------- 1. LIGHTING & QUALITY ANALYSIS ----------------

def analyze_lighting(image):
    """
    Analyzes optical capture quality:
    - Brightness: mean grayscale intensity (ideal: 55 - 225)
    - Contrast: standard deviation of grayscale intensity (ideal: >= 22)
    - Glare: percentage of saturated specular reflection pixels (>= 250) (destructive when contrast < 30)
    - Sharpness / Blur: Laplacian variance (ideal: >= 50)

    Returns dict with metrics, status (GOOD, POOR, TOO_BRIGHT, GLARE, LOW_CONTRAST, BLURRED),
    quality_verdict (PASS or REVIEW), and an explanatory message.
    """
    if image is None:
        return {
            "status": "INVALID",
            "quality_verdict": "REVIEW",
            "brightness": 0.0,
            "contrast": 0.0,
            "glare_pct": 0.0,
            "sharpness": 0.0,
            "message": "No valid image provided for lighting analysis."
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    glare_pct = float(np.mean(gray >= 250) * 100)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    issues = []
    status = "GOOD"
    quality_verdict = "PASS"

    # Blur detection
    if laplacian_var < 50:
        issues.append(f"Image is blurred / out of focus (sharpness index: {laplacian_var:.1f} < 50).")
        status = "BLURRED"
        quality_verdict = "REVIEW"

    # Underexposure
    if brightness < 55:
        issues.append(f"Image is underexposed / too dark (brightness: {brightness:.1f} < 55).")
        status = "POOR"
        quality_verdict = "REVIEW"

    # True glare / washed-out overexposure
    elif (brightness > 225 and contrast < 28) or (glare_pct > 35 and contrast < 30):
        issues.append(f"Excessive specular glare / washed-out exposure detected (glare: {glare_pct:.1f}%, contrast: {contrast:.1f}).")
        status = "GLARE" if glare_pct > 35 else "TOO_BRIGHT"
        quality_verdict = "REVIEW"

    # Low optical contrast
    elif contrast < 20:
        issues.append(f"Low optical contrast detected (contrast: {contrast:.1f} < 20).")
        if status == "GOOD":
            status = "LOW_CONTRAST"
        quality_verdict = "REVIEW"

    if quality_verdict == "PASS":
        message = "Optimal lighting and optical sharpness detected."
    else:
        message = " ".join(issues) + " Image capture quality requires manual review or re-capture."

    return {
        "status": status,
        "quality_verdict": quality_verdict,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "glare_pct": round(glare_pct, 1),
        "sharpness": round(laplacian_var, 1),
        "message": message
    }


# ---------------- 2. QR CODE DETECTION ----------------

def detect_qr(image):
    """
    Detects and decodes QR codes using OpenCV's QRCodeDetector.
    Uses multi-pass enhancement (standard, CLAHE, Otsu) to handle varying camera exposures.

    Returns:
    - qr_status: PASS or FAIL
    - qr_data: Decoded string or None
    - points: Corner coordinates or None
    - annotated_image: Image with bounding box overlay
    - message: Descriptive status message
    """
    if image is None:
        return "FAIL", None, None, None, "No image provided for QR detection."

    annotated = image.copy()
    detector = cv2.QRCodeDetector()

    # Pass 1: Raw image
    qr_data, points, _ = detector.detectAndDecode(image)

    # Pass 2: CLAHE enhanced if raw failed
    if not qr_data:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        qr_data, points, _ = detector.detectAndDecode(enhanced)

    # Pass 3: Otsu threshold fallback
    if not qr_data:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        qr_data, points, _ = detector.detectAndDecode(otsu)

    if qr_data and qr_data.strip():
        qr_data = qr_data.strip()
        # Draw bounding polygon if points are available
        if points is not None and len(points) > 0:
            pts = points.astype(int).reshape(-1, 2)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            # Add label tag near QR
            top_left = pts[0]
            cv2.putText(
                annotated,
                f"QR: {qr_data}",
                (max(10, top_left[0]), max(25, top_left[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        return "PASS", qr_data, points, annotated, f"QR code decoded: '{qr_data}'"

    return "FAIL", None, None, annotated, "QR code not detected or unreadable on product label."


# ---------------- 3. DATABASE VALIDATION ----------------

def validate_product_in_database(product_id):
    """
    Validates product ID against MySQL database:
    - Confirms product exists in database registry
    - Verifies manufacturing and expiry dates
    - Flags expired products as FAIL (Critical failure)

    Returns:
    - db_status: PASS or FAIL
    - product_record: dict of product fields or None
    - message: Detailed explanation
    """
    if not product_id:
        return "FAIL", None, "Missing Product ID from QR code."

    product = get_product(product_id)
    if not product:
        return (
            "FAIL",
            None,
            f"UNKNOWN PRODUCT: Product ID '{product_id}' is not registered in the central database."
        )

    # Verify expiration date
    exp_date = product.get("expiry_date")
    today = date.today()

    if isinstance(exp_date, str):
        try:
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
        except Exception:
            pass

    if exp_date and isinstance(exp_date, (date, datetime)):
        if isinstance(exp_date, datetime):
            exp_date = exp_date.date()

        if exp_date < today:
            days_expired = (today - exp_date).days
            return (
                "FAIL",
                product,
                f"CRITICAL: Product EXPIRED on {exp_date.strftime('%d-%m-%Y')} ({days_expired} days ago). Quality check rejected."
            )

    return "PASS", product, f"Product '{product.get('product_name')}' validated in database. Active and within expiry."


# ---------------- 4. OCR TEXT VERIFICATION ----------------

def normalize_text(text):
    """Normalizes alphanumeric text for robust OCR comparison."""
    if not text:
        return ""
    # Standardize common OCR substitutions (0 vs O)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(text).upper())
    return cleaned


def fuzzy_match(expected, actual):
    """Returns similarity ratio (0.0 - 1.0) between expected and actual text."""
    norm_exp = normalize_text(expected)
    norm_act = normalize_text(actual)
    if not norm_exp:
        return 1.0
    if norm_exp in norm_act:
        return 1.0
    return SequenceMatcher(None, norm_exp, norm_act).ratio()


def verify_ocr_text(image, product):
    """
    Extracts text via Tesseract OCR and performs field-by-field verification:
    - Product Name
    - Batch Number
    - Manufacturing Date
    - Expiry Date

    Returns:
    - text_status: PASS, REVIEW, or FAIL
    - extracted_raw_text: full OCR text
    - field_results: dict detailing verification of each field
    - message: Clear audit remarks
    """
    if image is None:
        return "REVIEW", "", {}, "No image provided for OCR."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Multi-step preprocessing for optimal character recognition
    enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.GaussianBlur(enlarged, (3, 3), 0)
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    try:
        raw_ocr = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
    except Exception as e:
        # Fallback without custom config if error
        try:
            raw_ocr = pytesseract.image_to_string(gray)
        except Exception as inner_e:
            return "REVIEW", "", {}, f"OCR engine error: {inner_e}"

    if not raw_ocr or len(raw_ocr.strip()) < 10:
        return (
            "REVIEW",
            raw_ocr or "",
            {},
            "Insufficient legible text extracted by OCR. Incomplete or obstructed label."
        )

    norm_ocr = normalize_text(raw_ocr)
    # Check for O vs 0 normalization
    norm_ocr_zero_swapped = norm_ocr.replace("O", "0")

    field_results = {}
    mismatches = []
    uncertainties = []

    # 1. Product Name check
    expected_name = str(product.get("product_name", ""))
    norm_expected_name = normalize_text(expected_name)
    if norm_expected_name in norm_ocr:
        field_results["Product Name"] = {"status": "MATCH", "expected": expected_name, "found": True}
    else:
        # Check similarity
        sim = fuzzy_match(expected_name, norm_ocr)
        if sim >= 0.85:
            field_results["Product Name"] = {"status": "MATCH (Fuzzy)", "expected": expected_name, "found": True}
        elif "FAKE" in norm_ocr or "COUNTERFEIT" in norm_ocr or (len(norm_ocr) > 50 and sim < 0.40):
            field_results["Product Name"] = {"status": "MISMATCH", "expected": expected_name, "found": False}
            mismatches.append(f"Product Name mismatch (expected '{expected_name}')")
        else:
            field_results["Product Name"] = {"status": "UNCERTAIN", "expected": expected_name, "found": False}
            uncertainties.append(f"Product Name unclear in OCR ({sim*100:.0f}% match)")

    # 2. Batch Number check
    expected_batch = str(product.get("batch_number", ""))
    norm_expected_batch = normalize_text(expected_batch)
    norm_batch_zero = norm_expected_batch.replace("O", "0")

    if (norm_expected_batch in norm_ocr) or (norm_batch_zero in norm_ocr_zero_swapped):
        field_results["Batch Number"] = {"status": "MATCH", "expected": expected_batch, "found": True}
    else:
        sim = fuzzy_match(expected_batch, norm_ocr)
        if sim >= 0.80:
            field_results["Batch Number"] = {"status": "MATCH (Fuzzy)", "expected": expected_batch, "found": True}
        else:
            uncertainties.append(f"Batch Number '{expected_batch}' not clearly verified")
            field_results["Batch Number"] = {"status": "UNCERTAIN", "expected": expected_batch, "found": False}

    # 3. Date Checks (if available)
    for date_key, label in [("manufacturing_date", "Mfg Date"), ("expiry_date", "Exp Date")]:
        dt_val = product.get(date_key)
        if dt_val:
            if isinstance(dt_val, (date, datetime)):
                date_str1 = dt_val.strftime("%d-%m-%Y")
                date_str2 = dt_val.strftime("%d/%m/%Y")
                date_str3 = dt_val.strftime("%Y-%m-%d")
            else:
                date_str1 = str(dt_val)
                date_str2 = date_str1
                date_str3 = date_str1

            match_found = any(
                normalize_text(ds) in norm_ocr for ds in [date_str1, date_str2, date_str3]
            )
            field_results[label] = {
                "status": "MATCH" if match_found else "PARTIAL",
                "expected": date_str1,
                "found": match_found
            }

    # Decision logic for OCR
    if mismatches:
        text_status = "FAIL"
        message = f"CRITICAL TEXT MISMATCH: {'; '.join(mismatches)}."
    elif uncertainties:
        text_status = "REVIEW"
        message = f"Uncertain OCR readings: {'; '.join(uncertainties)}. Manual review required."
    else:
        text_status = "PASS"
        message = "All critical printed label text matches registered database records."

    return text_status, raw_ocr, field_results, message


# ---------------- 5. ORB APPEARANCE VERIFICATION ----------------

def get_reference_template(reference_path="captured_images/sample_product_label.png"):
    """Finds reference image across standard paths."""
    candidates = [
        reference_path,
        "sample_product_label.png",
        "captured_images/sample_product_label.png",
        Path(__file__).parent / "captured_images" / "sample_product_label.png",
        Path(__file__).parent / "sample_product_label.png",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img, str(p)
    return None, None


def verify_appearance_orb(image, reference_path="captured_images/sample_product_label.png"):
    """
    Performs industrial appearance verification using ORB (Oriented FAST and Rotated BRIEF)
    and RANSAC Homography geometric alignment:
    - Detects keypoints and descriptors
    - Performs KNN feature matching with Lowe's ratio test (0.75)
    - Computes RANSAC Homography to test spatial layout consistency
    - Evaluates good matches, inlier count, and alignment ratio
    - Evaluates visual pixel difference on aligned label

    Returns:
    - appearance_status: PASS, REVIEW, or FAIL
    - metrics: dict of matches, inliers, alignment_ratio, pixel_difference
    - match_vis: visual image with matching keypoint rays/overlay
    - remarks: Detailed audit justification
    """
    reference, resolved_path = get_reference_template(reference_path)
    if reference is None:
        return (
            "REVIEW",
            {"matches": 0, "inliers": 0, "alignment_score": 0.0, "difference_pct": 0.0},
            None,
            "Reference label template not found on system. Manual inspection required."
        )

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cap_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lighting equalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_norm = clahe.apply(ref_gray)
    cap_norm = clahe.apply(cap_gray)

    # ORB feature detection
    orb = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=10
    )

    kp_ref, des_ref = orb.detectAndCompute(ref_norm, None)
    kp_cap, des_cap = orb.detectAndCompute(cap_norm, None)

    if des_ref is None or des_cap is None or len(des_ref) < 10 or len(des_cap) < 10:
        return (
            "REVIEW",
            {"matches": 0, "inliers": 0, "alignment_score": 0.0, "difference_pct": 0.0},
            None,
            "Insufficient optical features detected. Image may be severely degraded or blank."
        )

    # Feature matching via BFMatcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_cap, des_ref, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Appearance check if matches are too low
    if len(good_matches) < 6:
        # Check if the image had plenty of features (indicating an alien/completely wrong object)
        if len(kp_cap) > 100:
            return (
                "FAIL",
                {"matches": len(good_matches), "inliers": 0, "alignment_score": 0.0, "difference_pct": 100.0},
                None,
                f"CONFIRMED APPEARANCE MISMATCH: Only {len(good_matches)} feature matches found against reference label."
            )
        return (
            "REVIEW",
            {"matches": len(good_matches), "inliers": 0, "alignment_score": 0.0, "difference_pct": 0.0},
            None,
            f"Only {len(good_matches)} matching features found. Borderline visibility. Manual review required."
        )

    # Geometric alignment via Homography with RANSAC
    src_pts = np.float32([kp_cap[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if inlier_mask is None:
        return (
            "REVIEW",
            {"matches": len(good_matches), "inliers": 0, "alignment_score": 0.0, "difference_pct": 0.0},
            None,
            "Layout geometric alignment failed RANSAC consistency. Manual review required."
        )

    inliers = int(inlier_mask.sum())
    match_ratio = (inliers / len(good_matches)) if good_matches else 0.0

    # Draw top matches visualization
    draw_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
    # Pick sample matches to avoid clutter
    sample_matches = draw_matches[:40]
    vis_img = cv2.drawMatches(
        image, kp_cap, reference, kp_ref, sample_matches, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Compute pixel difference on geometrically aligned image
    h_ref, w_ref = ref_gray.shape
    try:
        warped_to_ref = cv2.warpPerspective(cap_norm, homography, (w_ref, h_ref))
        diff = cv2.absdiff(ref_norm, warped_to_ref)
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        diff_pct = (changed_pixels / thresh.size) * 100
    except Exception:
        diff_pct = 0.0

    metrics = {
        "matches": len(good_matches),
        "inliers": inliers,
        "alignment_score": round(match_ratio * 100, 1),
        "difference_pct": round(diff_pct, 2)
    }

    remarks = (
        f"Matching features: {len(good_matches)}; "
        f"aligned features: {inliers}; "
        f"alignment score: {match_ratio * 100:.1f}%; "
        f"visual diff: {diff_pct:.2f}%."
    )

    # Strict Decision Thresholds:
    # PASS: Inliers >= 15 and Alignment ratio >= 30%
    # REVIEW: Inliers between 6 and 14, or alignment ratio < 30%
    # FAIL: Inliers < 6 (with >100 kp) or confirmed layout defect
    if inliers >= 15 and match_ratio >= 0.30:
        return "PASS", metrics, vis_img, remarks + " Label layout matches authentic reference."
    elif inliers >= 6:
        return "REVIEW", metrics, vis_img, remarks + " Borderline layout alignment. Operator review advised."
    else:
        return "FAIL", metrics, vis_img, remarks + " Unacceptable layout mismatch. Rejected."


# ---------------- 6. INDUSTRIAL DECISION ENGINE (PHASE 5) ----------------

def evaluate_inspection(lighting_res, qr_res, db_res, ocr_res, appearance_res):
    """
    Implements consistent, strict industrial PASS/REVIEW/FAIL decision logic:
    - PASS only when all required checks pass.
    - REVIEW for uncertain OCR, poor image quality, uncertain appearance, or incomplete information.
    - FAIL for invalid/missing QR, unknown product, expired product, confirmed critical text mismatch,
      or confirmed appearance mismatch.
    - CRITICAL FAILURES STRICTLY OVERRIDE OTHER RESULTS.
    - Uncertainty is NEVER treated as failure or approval.

    Returns:
    - final_result: "PASS" | "REVIEW" | "FAIL"
    - audit_summary: Comprehensive explanation of decisions
    """
    failures = []
    reviews = []

    # 1. Optical capture check
    if lighting_res.get("quality_verdict") == "REVIEW":
        reviews.append(f"Image Quality: {lighting_res.get('message')}")

    # 2. QR Code check
    qr_status = qr_res.get("status")
    if qr_status == "FAIL":
        failures.append(f"QR Code: {qr_res.get('message')}")
    elif qr_status == "REVIEW":
        reviews.append(f"QR Code: {qr_res.get('message')}")

    # 3. Database check
    db_status = db_res.get("status")
    if db_status == "FAIL":
        failures.append(f"Database Registry: {db_res.get('message')}")
    elif db_status == "REVIEW":
        reviews.append(f"Database Registry: {db_res.get('message')}")

    # 4. OCR Text check
    text_status = ocr_res.get("status")
    if text_status == "FAIL":
        failures.append(f"OCR Verification: {ocr_res.get('message')}")
    elif text_status == "REVIEW":
        reviews.append(f"OCR Verification: {ocr_res.get('message')}")

    # 5. Appearance check
    app_status = appearance_res.get("status")
    if app_status == "FAIL":
        failures.append(f"Appearance Verification: {appearance_res.get('message')}")
    elif app_status == "REVIEW":
        reviews.append(f"Appearance Verification: {appearance_res.get('message')}")

    # Strict resolution hierarchy:
    # Critical failures override all other stages
    if failures:
        final_result = "FAIL"
        audit_summary = "CRITICAL DEFECT DETECTED: " + " | ".join(failures)
    elif reviews:
        final_result = "REVIEW"
        audit_summary = "MANUAL OPERATOR REVIEW REQUIRED: " + " | ".join(reviews)
    else:
        final_result = "PASS"
        audit_summary = (
            "ALL CHECKS PASSED: Authentic label verified. "
            f"QR: PASS, DB: PASS, OCR: PASS, Appearance: PASS ({appearance_res.get('metrics', {}).get('inliers', 0)} inliers)."
        )

    return final_result, audit_summary
