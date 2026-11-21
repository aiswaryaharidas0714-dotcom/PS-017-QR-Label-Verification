# Automated QR Labeling and Appearance Verification System
### Sakthi Hackathon 2.0 — Problem Statement: PS-017

An end-to-end industrial computer-vision and database-backed quality assurance system for inspecting manufactured product labels. The system performs real-time optical quality analysis, OpenCV QR code localization and decoding, MySQL database validation, Tesseract OCR text verification, and ORB feature-based geometric appearance verification.

---

## 🏛️ System Architecture & Inspection Pipeline

```
[ Camera Capture / Image Upload ]
               │
               ▼
   [ Stage 1: Lighting & Optical Quality Analysis ]
   - Mean Brightness (55 - 225)
   - Contrast Std Dev (>= 20)
   - Glare Saturation (< 35%)
   - Laplacian Sharpness (>= 50)
               │
               ▼
   [ Stage 2: QR Code Detection & Decoding ]
   - OpenCV QRCodeDetector with multi-pass CLAHE / Otsu fallback
   - Product ID extraction & bounding polygon annotation
               │
               ▼
   [ Stage 3: MySQL Database Validation ]
   - Parameterized query against `products` table
   - Expiration date verification (`expiry_date >= today`)
               │
               ▼
   [ Stage 4: Tesseract OCR Text Verification ]
   - Adaptive thresholding & noise reduction
   - Extracted text vs database fields (Product Name, Batch Number, Dates)
   - Fuzzy similarity and tamper detection
               │
               ▼
   [ Stage 5: ORB Appearance Verification ]
   - Keypoint detection & KNN BFMatcher with Lowe's ratio test (0.75)
   - RANSAC Homography geometric alignment
   - Inlier count, alignment ratio, and aligned pixel difference
               │
               ▼
   [ Stage 6: Industrial PASS / REVIEW / FAIL Decision Matrix ]
   - PASS: All stages verified compliant
   - REVIEW: Quality uncertainty or incomplete data
   - FAIL: Critical failure (Missing QR, unknown/expired product, text mismatch, tampered layout)
               │
               ▼
   [ MySQL Persistence & Analytics ]
   - Stored in `inspection_history`
   - Real-time Dashboard KPIs, Charts & CSV Audit Export
```

---

## 🎯 Consistent Decision Logic & Thresholds (Phase 5)

| Inspection Stage | Metric / Parameter | PASS Threshold | REVIEW Threshold | FAIL Threshold (Critical) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Lighting & Optics** | Brightness, Contrast, Sharpness | Brightness: 55–225<br>Contrast: &ge; 20<br>Sharpness: &ge; 50 | Underexposed (&lt;55)<br>Glare (&gt;35%)<br>Blur / Defocus (&lt;50) | Severe physical obstruction |
| **2. QR Code** | Detection & Decoding | Valid string decoded matching DB format | Boundary partially clipped | Missing, damaged, or unreadable QR |
| **3. Database Registry** | MySQL lookup & Expiration | Registered product active and within expiry date | — | Unknown Product ID<br>Expired Product (`expiry_date < today`) |
| **4. OCR Text** | Tesseract String Verification | Product Name & Batch match registered record (&ge;85% fuzzy) | Borderline OCR similarity<br>Low character count (&lt;15 chars) | Confirmed critical text mismatch<br>(e.g. counterfeit title "FAKE PRODUCT") |
| **5. ORB Appearance** | Keypoints & RANSAC Homography | Inliers &ge; 15<br>Alignment score &ge; 30%<br>Aligned diff &le; 0.5% | Inliers between 6 and 14<br>Template missing | Inliers &lt; 6 with adequate features<br>Confirmed tampered layout |
| **Overall Verdict** | **Consolidated Decision** | **ALL stages PASS** | **Any stage REVIEW and no FAIL** | **ANY critical defect strictly overrides all** |

---

## 🚀 Key Features

1. **Industrial Responsive Dashboard**: Dark Navy (`#0B192C`, `#1E293B`), white, and slate palette with high-contrast PASS (`#10B981`), REVIEW (`#F59E0B`), and FAIL (`#EF4444`) status badges.
2. **Dual Capture Pipeline**: Real-time webcam capture (`st.camera_input`) as primary mode, with high-resolution image upload (`st.file_uploader`) as secondary QA mode.
3. **Live Executive Dashboard**: All metrics (Total Inspections, PASS, REVIEW, FAIL, Yield Rate %, Status donut, 7-day trend chart, and recent feed) calculated dynamically from live MySQL records.
4. **Audit History & Search**: Search by Product ID or remarks, filter by verdict, view comprehensive inspection modal/drawer, and download inspection reports as CSV.
5. **Product Master Registry**: View products, register new items with validation, update existing records, and generate/download standardized printable QR labels.
6. **Robust Security & Reliability**: Parameterized SQL (`%s`), credentials managed via Streamlit secrets (`.streamlit/secrets.toml`), file validation, and fault-tolerant database handling.

---

## 💻 Local Setup & Run Instructions

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12+ (tested on Python 3.14)
- Tesseract OCR:
  - **Windows**: Install [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
  - **Linux / Ubuntu**: `sudo apt-get install -y tesseract-ocr`
  - **macOS**: `brew install tesseract`

### 2. Clone and Setup Environment
```bash
git clone <repository_url>
cd PS-017-QR-Label-Verification

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Database Credentials
Create `.streamlit/secrets.toml`:
```toml
[mysql]
host = "your-mysql-host.aivencloud.com"
port = 18872
user = "avnadmin"
password = "your-secure-password"
database = "defaultdb"
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing Suite (Phase 10)

A comprehensive automated test suite (`test_suite.py`) verifies all 12 operational scenarios specified:

```bash
python test_suite.py
```

### Actual Test Outcomes:
```
================================================================================
PS-017 INDUSTRIAL AUTOMATED INSPECTION TEST SUITE
================================================================================

[TEST 1] Testing Valid Product Label (sample_product_label.png)...
 -> Outcome: PASS | Summary: ALL CHECKS PASSED: Authentic label verified. QR: PASS, DB: PASS, OCR: PASS, Appearance: PASS (1439 inliers).

[TEST 2] Testing Unknown Product ID ('PROD_UNKNOWN_999')...
 -> Outcome: FAIL | DB Status: FAIL | Reason: UNKNOWN PRODUCT: Product ID 'PROD_UNKNOWN_999' is not registered in the central database.

[TEST 3] Testing Missing / Unreadable QR Code...
 -> Outcome: FAIL | QR Status: FAIL | Reason: QR code not detected or unreadable on product label.

[TEST 4] Testing Tampered Label Name ('FAKE PRODUCTLABEL')...
 -> Outcome: FAIL | OCR Status: FAIL | Reason: CRITICAL TEXT MISMATCH: Product Name mismatch (expected 'Herbal Shampoo').

[TEST 5] Testing Expired Product Detection...
 -> Outcome: FAIL | DB Status: FAIL | Reason: CRITICAL: Product EXPIRED on 07-10-2026 (45 days ago). Quality check rejected.

[TEST 6] Testing Blurred / Out-of-Focus Image...
 -> Outcome: REVIEW | Quality Status: BLURRED | Sharpness: 0.9 < 50

[TEST 7] Testing Poor Lighting (Underexposure & Glare)...
 -> Dark Image Outcome: POOR (Brightness: 35.7 < 55)
 -> Glare Image Outcome: GLARE (Glare: 98.4%, Contrast: 1.9 < 20)

[TEST 8] Testing Appearance Mismatch with Alien Pattern...
 -> Outcome: FAIL | Appearance Status: FAIL | Inliers: 5 < 6 | Reason: Unacceptable layout mismatch. Rejected.

[TEST 9] Testing Database Unavailable / Fault Injection Handling...
 -> Outcome: Handled gracefully without unhandled exceptions. Error message captured.

[TEST 10] Testing Saving Inspection History Record...
 -> Outcome: SUCCESS | Record created with Inspection ID: #13 in MySQL.

[TEST 11] Testing Retrieving Inspection History from MySQL...
 -> Outcome: SUCCESS | Retrieved recent records with all schema columns intact.

[TEST 12] Testing Dashboard Statistics from Live MySQL Records...
 -> Outcome: SUCCESS | Real Metrics: Total=13, PASS=6, REVIEW=5, FAIL=2, Pass Rate=46.2%

----------------------------------------------------------------------
Ran 12 tests in 14.199s
OK
```

---

## 🌐 Cloud Deployment Instructions (Streamlit Community Cloud)

1. Push your repository to GitHub.
2. In Streamlit Community Cloud:
   - Select your repository and set `Main file path` to `app.py`.
   - In **App Settings** &rarr; **Secrets**, paste your MySQL credentials block:
     ```toml
     [mysql]
     host = "your-host.aivencloud.com"
     port = 18872
     user = "avnadmin"
     password = "your-password"
     database = "defaultdb"
     ```
3. `packages.txt` automatically instructs Debian to install `tesseract-ocr`.
4. Deploy the application.

---

## ⚠️ Remaining Hardware Limitations & Practical Recommendations

1. **Webcam Optical Auto-Focus**: Fixed-focus low-resolution laptop webcams (< 720p) may produce blurred images if held closer than 15 cm. Recommended hardware: 1080p industrial USB inspection camera with macro autofocus or manual lens ring.
2. **Specular Reflections on Glossy / Foil Labels**: Highly reflective metallic or plastic foil labels can produce specular glare under direct overhead lighting. Recommended setup: Ring diffuser light or indirect diffused cross-lighting at 45-degree angles.
3. **Multi-Label Visual Framing**: If multiple product labels are visible in a single frame, the primary label closest to the center with the most prominent QR code will be prioritized. Operators should frame one label at a time.
