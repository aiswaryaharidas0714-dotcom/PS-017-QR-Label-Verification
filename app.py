import os
import io
from datetime import datetime, date
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

import database as db
import inspection_engine as ie

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Automated QR Labeling & Appearance Verification | PS-017",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- INDUSTRIAL THEME & STYLESHEET ----------------
st.markdown("""
<style>
    /* Dark Navy & Slate Industrial Palette */
    :root {
        --navy-dark: #0B192C;
        --navy-card: #1E293B;
        --navy-border: #334155;
        --slate-text: #94A3B8;
        --white-text: #F8FAFC;
        --pass-color: #10B981;
        --pass-bg: rgba(16, 185, 129, 0.12);
        --review-color: #F59E0B;
        --review-bg: rgba(245, 158, 11, 0.12);
        --fail-color: #EF4444;
        --fail-bg: rgba(239, 68, 68, 0.12);
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Top Brand Header */
    .industrial-header {
        background: linear-gradient(135deg, #0B192C 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .industrial-header h1 {
        color: #F8FAFC !important;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .industrial-header p {
        color: #94A3B8 !important;
        font-size: 0.88rem;
        margin: 0.35rem 0 0 0;
    }

    /* Metric KPI Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #64748B;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }

    /* Custom Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-pass {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .badge-review {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }
    .badge-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
    }

    /* Final Decision Banner */
    .verdict-banner-pass {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.18) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .verdict-banner-review {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.18) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .verdict-banner-fail {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.18) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
    }

    /* Stage Panel Cards */
    .stage-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .stage-card h4 {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
    }

    /* Diagnostic table text */
    .diag-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .diag-table td {
        padding: 0.45rem 0.6rem;
        border-bottom: 1px solid #334155;
    }
    .diag-label {
        color: #94A3B8;
        font-weight: 500;
        width: 38%;
    }
    .diag-val {
        color: #F8FAFC;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR NAVIGATION & SYSTEM STATUS ----------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0; text-align: center;">
        <div style="font-size: 1.6rem; font-weight: 800; color: #38BDF8; letter-spacing: -0.5px;">
            PS-017 INSPECTOR
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500;">
            Sakthi Hackathon 2.0 Industrial Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "NAVIGATION MODULES",
        [
            "🔬 Live Inspection",
            "📊 Executive Dashboard",
            "📜 Inspection History",
            "📦 Product Management",
            "📈 Analytics & Reports",
            "⚙️ System & Diagnostics"
        ],
        index=0
    )

    st.markdown("---")

    # Live Database Health Status Widget
    db_ok, db_ping = db.check_connection()
    if db_ok:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 6px; padding: 0.6rem; font-size: 0.8rem;">
            <div style="color: #10B981; font-weight: 700;">🟢 MySQL CONNECTED</div>
            <div style="color: #94A3B8; font-size: 0.72rem; margin-top: 2px;">Latency: {db_ping}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 6px; padding: 0.6rem; font-size: 0.8rem;">
            <div style="color: #EF4444; font-weight: 700;">🔴 DATABASE OFFLINE</div>
            <div style="color: #94A3B8; font-size: 0.72rem; margin-top: 2px;">{db_ping}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tesseract Status
    tess_avail = ie.TESSERACT_EXE is not None
    if tess_avail:
        st.markdown("""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38BDF8; border-radius: 6px; padding: 0.6rem; font-size: 0.8rem; margin-top: 0.5rem;">
            <div style="color: #38BDF8; font-weight: 700;">⚡ OCR ENGINE READY</div>
            <div style="color: #94A3B8; font-size: 0.72rem; margin-top: 2px;">Tesseract OCR Active</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #F59E0B; border-radius: 6px; padding: 0.6rem; font-size: 0.8rem; margin-top: 0.5rem;">
            <div style="color: #F59E0B; font-weight: 700;">⚠️ OCR ENGINE MISSING</div>
            <div style="color: #94A3B8; font-size: 0.72rem; margin-top: 2px;">Check Tesseract install</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 2rem; font-size: 0.72rem; color: #64748B; text-align: center;">
        PS-017 Automated QR & Appearance Verification System<br>
        Version 2.5 Industrial Release
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: LIVE INSPECTION (PHASE 4 & 5)
# ==============================================================================
if menu == "🔬 Live Inspection":
    st.markdown("""
    <div class="industrial-header">
        <h1>Industrial Product-Label Inspection Terminal</h1>
        <p>Real-time optical quality analysis, OpenCV QR decoding, MySQL database verification, Tesseract OCR reading and ORB geometric appearance matching.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input method selection: Webcam primary, Upload secondary
    col_input_sel, col_hint = st.columns([1.5, 2.5])
    with col_input_sel:
        input_mode = st.radio(
            "Acquisition Source",
            ["📸 Industrial Webcam (Primary)", "📁 Image File Upload (Backup / QA)"],
            horizontal=False
        )
    with col_hint:
        st.info(
            "**Operational Guidance**: Ensure the entire product label is framed within the acquisition window. "
            "Good lighting and minimal specular glare ensure high-yield verification."
        )

    image_source = None

    if "Webcam" in input_mode:
        st.markdown("##### 1. Live Optical Capture")
        image_source = st.camera_input("Capture Product Label via Webcam")
    else:
        st.markdown("##### 1. Select Inspection File")
        image_source = st.file_uploader(
            "Upload high-resolution label image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG. Max file size: 12MB."
        )

    if image_source is not None:
        try:
            # Decode image into OpenCV format
            file_bytes = np.asarray(bytearray(image_source.getvalue()), dtype=np.uint8)
            cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if cv_image is None:
                st.error("Error: Could not decode image format. Please capture or upload a valid image.")
            else:
                # Resize if oversized to prevent memory bloat while maintaining OCR clarity
                h, w = cv_image.shape[:2]
                if max(h, w) > 1600:
                    scale = 1600 / max(h, w)
                    cv_image = cv2.resize(cv_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                # Two-column layout: Left image preview, Right execution button & overview
                col_img, col_act = st.columns([1.3, 1])
                with col_img:
                    st.image(
                        cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB),
                        caption=f"Acquired Frame ({cv_image.shape[1]}x{cv_image.shape[0]} px)",
                        use_container_width=True
                    )
                with col_act:
                    st.markdown("""
                    <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem;">
                        <h4 style="margin: 0 0 0.75rem 0; color: #F8FAFC;">Acquisition Ready</h4>
                        <p style="color: #94A3B8; font-size: 0.85rem; margin: 0 0 1.25rem 0;">
                            Frame successfully staged. Click the button below to execute full automated verification pipeline across Optical, QR, MySQL, OCR, and ORB stages.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    run_inspection = st.button(
                        "⚡ EXECUTE INDUSTRIAL INSPECTION",
                        type="primary",
                        use_container_width=True
                    )

                if run_inspection:
                    with st.spinner("Executing 5-Stage Automated Inspection Pipeline..."):
                        # STAGE 1: Lighting & Optical Quality Analysis
                        light_res = ie.analyze_lighting(cv_image)

                        # STAGE 2: QR Code Detection
                        qr_status, qr_val, qr_points, qr_anno, qr_msg = ie.detect_qr(cv_image)

                        # STAGE 3: Database Validation
                        if qr_status == "PASS" and qr_val:
                            db_status, prod_record, db_msg = ie.validate_product_in_database(qr_val)
                        else:
                            db_status = "FAIL"
                            prod_record = None
                            db_msg = "Database validation skipped: QR code missing or unreadable."

                        # STAGE 4: OCR Text Verification
                        if db_status == "PASS" and prod_record:
                            ocr_status, ocr_raw, ocr_fields, ocr_msg = ie.verify_ocr_text(cv_image, prod_record)
                        else:
                            ocr_status = "REVIEW" if qr_status == "PASS" else "FAIL"
                            ocr_raw = ""
                            ocr_fields = {}
                            ocr_msg = "OCR text verification aborted: Registered product details unavailable."

                        # STAGE 5: ORB Appearance Verification
                        app_status, app_metrics, app_vis, app_msg = ie.verify_appearance_orb(cv_image)

                        # STAGE 6: Consistent Industrial Decision Matrix (Phase 5)
                        final_verdict, final_summary = ie.evaluate_inspection(
                            light_res,
                            {"status": qr_status, "message": qr_msg},
                            {"status": db_status, "message": db_msg},
                            {"status": ocr_status, "message": ocr_msg},
                            {"status": app_status, "metrics": app_metrics, "message": app_msg}
                        )

                        # Persist to MySQL Database (Phase 6 & 9)
                        saved_ok, saved_id_or_err = db.save_inspection(
                            product_id=qr_val if qr_val else "UNKNOWN",
                            qr_status=qr_status,
                            text_status=ocr_status,
                            appearance_status=app_status,
                            final_result=final_verdict,
                            remarks=f"{final_summary[:200]}"
                        )

                    # ---------------- VERDICT PRESENTATION ----------------
                    if final_verdict == "PASS":
                        st.markdown(f"""
                        <div class="verdict-banner-pass">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="status-badge badge-pass">VERDICT: PASS</span>
                                    <h2 style="color: #10B981; margin: 0.5rem 0 0.2rem 0; font-weight: 800;">AUTHENTIC PRODUCT LABEL CONFIRMED</h2>
                                    <div style="color: #E2E8F0; font-size: 0.95rem;">{final_summary}</div>
                                </div>
                                <div style="text-align: right; color: #94A3B8; font-size: 0.8rem;">
                                    Audit Record ID: #{saved_id_or_err if saved_ok else 'PENDING'}<br>
                                    Status: Quality Approved
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.toast("Inspection Complete: PASS Recorded!", icon="✅")

                    elif final_verdict == "FAIL":
                        st.markdown(f"""
                        <div class="verdict-banner-fail">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="status-badge badge-fail">VERDICT: FAIL</span>
                                    <h2 style="color: #EF4444; margin: 0.5rem 0 0.2rem 0; font-weight: 800;">INSPECTION REJECTED - CRITICAL DEFECT</h2>
                                    <div style="color: #E2E8F0; font-size: 0.95rem;">{final_summary}</div>
                                </div>
                                <div style="text-align: right; color: #94A3B8; font-size: 0.8rem;">
                                    Audit Record ID: #{saved_id_or_err if saved_ok else 'PENDING'}<br>
                                    Status: Quarantine Product
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.toast("Inspection Rejected: Critical Defect Recorded!", icon="🚨")

                    else:
                        st.markdown(f"""
                        <div class="verdict-banner-review">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="status-badge badge-review">VERDICT: REVIEW</span>
                                    <h2 style="color: #F59E0B; margin: 0.5rem 0 0.2rem 0; font-weight: 800;">OPERATOR REVIEW REQUIRED</h2>
                                    <div style="color: #E2E8F0; font-size: 0.95rem;">{final_summary}</div>
                                </div>
                                <div style="text-align: right; color: #94A3B8; font-size: 0.8rem;">
                                    Audit Record ID: #{saved_id_or_err if saved_ok else 'PENDING'}<br>
                                    Status: Manual Verification
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.toast("Inspection Warning: Operator Review Flagged!", icon="⚠️")

                    # ---------------- STAGE-BY-STAGE BREAKDOWN ----------------
                    st.markdown("### Detailed Verification Breakdown")

                    tab_stages = st.tabs([
                        "1. Lighting & Optics",
                        "2. QR Code Decoding",
                        "3. Database Record",
                        "4. OCR Text Analysis",
                        "5. ORB Appearance"
                    ])

                    # TAB 1: Lighting & Quality
                    with tab_stages[0]:
                        col_l1, col_l2 = st.columns([1, 1.2])
                        with col_l1:
                            st.markdown(f"""
                            <table class="diag-table">
                                <tr><td class="diag-label">Optical Status</td><td class="diag-val"><span class="status-badge {'badge-pass' if light_res['quality_verdict']=='PASS' else 'badge-review'}">{light_res['status']}</span></td></tr>
                                <tr><td class="diag-label">Mean Brightness</td><td class="diag-val">{light_res['brightness']} (Ideal: 55 - 225)</td></tr>
                                <tr><td class="diag-label">Optical Contrast</td><td class="diag-val">{light_res['contrast']} (Ideal: &ge; 22)</td></tr>
                                <tr><td class="diag-label">Specular Glare</td><td class="diag-val">{light_res['glare_pct']}% (Threshold: &lt; 35%)</td></tr>
                                <tr><td class="diag-label">Sharpness Index</td><td class="diag-val">{light_res['sharpness']} (Ideal: &ge; 50)</td></tr>
                            </table>
                            """, unsafe_allow_html=True)
                        with col_l2:
                            st.write("**Optical Quality Diagnosis:**")
                            st.info(light_res["message"])

                    # TAB 2: QR Code
                    with tab_stages[1]:
                        col_q1, col_q2 = st.columns([1, 1])
                        with col_q1:
                            st.markdown(f"""
                            <table class="diag-table">
                                <tr><td class="diag-label">QR Detection Status</td><td class="diag-val"><span class="status-badge {'badge-pass' if qr_status=='PASS' else 'badge-fail'}">{qr_status}</span></td></tr>
                                <tr><td class="diag-label">Decoded Product ID</td><td class="diag-val"><code>{qr_val or 'NOT DETECTED'}</code></td></tr>
                                <tr><td class="diag-label">Diagnostic Detail</td><td class="diag-val">{qr_msg}</td></tr>
                            </table>
                            """, unsafe_allow_html=True)
                        with col_q2:
                            if qr_anno is not None:
                                st.image(cv2.cvtColor(qr_anno, cv2.COLOR_BGR2RGB), caption="QR Detection & Localization Overlay", use_container_width=True)

                    # TAB 3: Database Validation
                    with tab_stages[2]:
                        if prod_record:
                            st.markdown(f"""
                            <table class="diag-table">
                                <tr><td class="diag-label">Database Validation</td><td class="diag-val"><span class="status-badge {'badge-pass' if db_status=='PASS' else 'badge-fail'}">{db_status}</span></td></tr>
                                <tr><td class="diag-label">Product ID</td><td class="diag-val"><code>{prod_record.get('product_id')}</code></td></tr>
                                <tr><td class="diag-label">Product Name</td><td class="diag-val">{prod_record.get('product_name')}</td></tr>
                                <tr><td class="diag-label">Batch Number</td><td class="diag-val">{prod_record.get('batch_number')}</td></tr>
                                <tr><td class="diag-label">Manufacturing Date</td><td class="diag-val">{prod_record.get('manufacturing_date')}</td></tr>
                                <tr><td class="diag-label">Expiry Date</td><td class="diag-val">{prod_record.get('expiry_date')}</td></tr>
                                <tr><td class="diag-label">Validation Audit</td><td class="diag-val">{db_msg}</td></tr>
                            </table>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(db_msg)

                    # TAB 4: OCR Text Verification
                    with tab_stages[3]:
                        col_o1, col_o2 = st.columns([1.1, 1])
                        with col_o1:
                            st.markdown(f"""
                            <table class="diag-table">
                                <tr><td class="diag-label">OCR Text Status</td><td class="diag-val"><span class="status-badge {'badge-pass' if ocr_status=='PASS' else ('badge-fail' if ocr_status=='FAIL' else 'badge-review')}">{ocr_status}</span></td></tr>
                                <tr><td class="diag-label">Evaluation Summary</td><td class="diag-val">{ocr_msg}</td></tr>
                            </table>
                            """, unsafe_allow_html=True)

                            if ocr_fields:
                                st.markdown("##### Field-by-Field Verification")
                                for f_name, f_info in ocr_fields.items():
                                    st.write(f"- **{f_name}**: `{f_info.get('expected')}` &rarr; **{f_info.get('status')}**")
                        with col_o2:
                            st.write("**Extracted OCR Raw Text:**")
                            st.text_area(
                                "Raw Tesseract String",
                                value=ocr_raw if ocr_raw.strip() else "(No legible characters extracted)",
                                height=160,
                                disabled=True
                            )

                    # TAB 5: ORB Appearance Verification
                    with tab_stages[4]:
                        col_a1, col_a2 = st.columns([1, 1.2])
                        with col_a1:
                            st.markdown(f"""
                            <table class="diag-table">
                                <tr><td class="diag-label">Appearance Status</td><td class="diag-val"><span class="status-badge {'badge-pass' if app_status=='PASS' else ('badge-fail' if app_status=='FAIL' else 'badge-review')}">{app_status}</span></td></tr>
                                <tr><td class="diag-label">Matching Features</td><td class="diag-val">{app_metrics.get('matches', 0)} keypoints</td></tr>
                                <tr><td class="diag-label">RANSAC Inliers</td><td class="diag-val">{app_metrics.get('inliers', 0)} aligned (Min: 15)</td></tr>
                                <tr><td class="diag-label">Alignment Score</td><td class="diag-val">{app_metrics.get('alignment_score', 0)}% (Min: 30%)</td></tr>
                                <tr><td class="diag-label">Aligned Pixel Diff</td><td class="diag-val">{app_metrics.get('difference_pct', 0)}%</td></tr>
                                <tr><td class="diag-label">Visual Audit</td><td class="diag-val">{app_msg}</td></tr>
                            </table>
                            """, unsafe_allow_html=True)
                        with col_a2:
                            if app_vis is not None:
                                st.image(
                                    cv2.cvtColor(app_vis, cv2.COLOR_BGR2RGB),
                                    caption="ORB Keypoint Correspondence & RANSAC Inlier Vectors",
                                    use_container_width=True
                                )

        except Exception as ex:
            st.error(f"Inspection Engine Exception: {ex}")


# ==============================================================================
# MODULE 2: EXECUTIVE DASHBOARD (PHASE 3)
# ==============================================================================
elif menu == "📊 Executive Dashboard":
    st.markdown("""
    <div class="industrial-header">
        <h1>Production Quality Executive Dashboard</h1>
        <p>Live metrics, inspection yield rates, and defect distribution queried directly from central MySQL database.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = db.get_dashboard_metrics()

    # KPI Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Inspections</div>
            <div class="metric-value">{metrics['total']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #10B981;">
            <div class="metric-label" style="color: #10B981;">Total PASS</div>
            <div class="metric-value" style="color: #10B981;">{metrics['pass']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #F59E0B;">
            <div class="metric-label" style="color: #F59E0B;">Total REVIEW</div>
            <div class="metric-value" style="color: #F59E0B;">{metrics['review']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #EF4444;">
            <div class="metric-label" style="color: #EF4444;">Total FAIL</div>
            <div class="metric-value" style="color: #EF4444;">{metrics['fail']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #38BDF8;">
            <div class="metric-label" style="color: #38BDF8;">Production Yield</div>
            <div class="metric-value" style="color: #38BDF8;">{metrics['pass_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    # Visual Analytics Charts
    col_chart1, col_chart2 = st.columns([1.1, 1.4])

    with col_chart1:
        st.markdown("##### Inspection Status Distribution")
        status_df = pd.DataFrame([
            {"Status": "PASS", "Count": metrics["pass"], "Color": "#10B981"},
            {"Status": "REVIEW", "Count": metrics["review"], "Color": "#F59E0B"},
            {"Status": "FAIL", "Count": metrics["fail"], "Color": "#EF4444"},
        ])

        if metrics["total"] > 0:
            donut_chart = alt.Chart(status_df).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Status",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["PASS", "REVIEW", "FAIL"],
                        range=["#10B981", "#F59E0B", "#EF4444"]
                    )
                ),
                tooltip=["Status", "Count"]
            ).properties(height=280)
            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No inspection data available for chart visualization.")

    with col_chart2:
        st.markdown("##### Production Daily Trend (Past 7 Days)")
        if metrics.get("daily_trends"):
            trend_df = pd.DataFrame(metrics["daily_trends"])
            melted_trend = trend_df.melt(
                id_vars=["date"],
                value_vars=["pass_count", "review_count", "fail_count"],
                var_name="Category",
                value_name="Count"
            )
            melted_trend["Category"] = melted_trend["Category"].map({
                "pass_count": "PASS",
                "review_count": "REVIEW",
                "fail_count": "FAIL"
            })
            bar_chart = alt.Chart(melted_trend).mark_bar().encode(
                x=alt.X("date:O", title="Date"),
                y=alt.Y("Count:Q", title="Inspection Count"),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(
                        domain=["PASS", "REVIEW", "FAIL"],
                        range=["#10B981", "#F59E0B", "#EF4444"]
                    )
                ),
                tooltip=["date", "Category", "Count"]
            ).properties(height=280)
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No daily history records recorded yet.")

    # Recent Activity Feed
    st.markdown("##### Recent Production Inspections")
    if metrics["recent_inspections"]:
        recent_df = pd.DataFrame(metrics["recent_inspections"])
        recent_df.rename(columns={
            "inspection_id": "ID",
            "product_id": "Product ID",
            "inspection_date": "Timestamp",
            "qr_status": "QR",
            "text_status": "OCR",
            "appearance_status": "ORB",
            "final_result": "Verdict",
            "remarks": "Audit Remarks"
        }, inplace=True)
        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent inspection activity logged.")


# ==============================================================================
# MODULE 3: INSPECTION HISTORY (PHASE 6)
# ==============================================================================
elif menu == "📜 Inspection History":
    st.markdown("""
    <div class="industrial-header">
        <h1>Inspection History & Audit Trail</h1>
        <p>Comprehensive audit log of all physical inspections stored securely in MySQL. Filter, search, and inspect complete stage decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Search & Filter Controls
    col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.2, 1, 0.8])
    with col_f1:
        search_query = st.text_input("🔍 Search by Product ID or Remarks", placeholder="e.g. PROD001 or expired")
    with col_f2:
        status_filter = st.selectbox("Filter Verdict", ["ALL", "PASS", "REVIEW", "FAIL"])
    with col_f3:
        record_limit = st.selectbox("Record Limit", [25, 50, 100, 250], index=1)
    with col_f4:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)

    history_data = db.get_inspection_history(
        limit=record_limit,
        status_filter=status_filter,
        search_query=search_query
    )

    if history_data:
        df_history = pd.DataFrame(history_data)

        # Download CSV button (Phase 8)
        csv_buffer = io.StringIO()
        df_history.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download History CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"inspection_audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Exports filtered inspection audit records to standard CSV format."
        )

        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "inspection_id": st.column_config.NumberColumn("Inspection ID", format="%d"),
                "product_id": "Product ID",
                "inspection_date": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "qr_status": "QR Code",
                "text_status": "OCR Text",
                "appearance_status": "ORB Appearance",
                "final_result": "Final Verdict",
                "remarks": "Diagnostic Audit Remarks"
            }
        )

        st.markdown("---")
        # Detailed Inspection Record Viewer
        st.markdown("##### Detailed Inspection Record Inspector")
        inspect_ids = [row["inspection_id"] for row in history_data]
        selected_id = st.selectbox("Select Inspection ID for Detailed View", inspect_ids)

        selected_row = next((r for r in history_data if r["inspection_id"] == selected_id), None)
        if selected_row:
            col_d1, col_d2 = st.columns([1, 1.5])
            with col_d1:
                st.markdown(f"""
                <table class="diag-table">
                    <tr><td class="diag-label">Inspection ID</td><td class="diag-val">#{selected_row['inspection_id']}</td></tr>
                    <tr><td class="diag-label">Product ID</td><td class="diag-val"><code>{selected_row['product_id']}</code></td></tr>
                    <tr><td class="diag-label">Timestamp</td><td class="diag-val">{selected_row['inspection_date']}</td></tr>
                    <tr><td class="diag-label">Final Verdict</td><td class="diag-val"><span class="status-badge {'badge-pass' if selected_row['final_result']=='PASS' else ('badge-fail' if selected_row['final_result']=='FAIL' else 'badge-review')}">{selected_row['final_result']}</span></td></tr>
                </table>
                """, unsafe_allow_html=True)
            with col_d2:
                st.markdown(f"""
                <table class="diag-table">
                    <tr><td class="diag-label">QR Status</td><td class="diag-val">{selected_row['qr_status']}</td></tr>
                    <tr><td class="diag-label">OCR Text Status</td><td class="diag-val">{selected_row['text_status']}</td></tr>
                    <tr><td class="diag-label">Appearance Status</td><td class="diag-val">{selected_row['appearance_status']}</td></tr>
                    <tr><td class="diag-label">Audit Remarks</td><td class="diag-val">{selected_row['remarks']}</td></tr>
                </table>
                """, unsafe_allow_html=True)

    else:
        st.info("No inspection records match the current filter criteria.")


# ==============================================================================
# MODULE 4: PRODUCT MANAGEMENT (PHASE 7)
# ==============================================================================
elif menu == "📦 Product Management":
    st.markdown("""
    <div class="industrial-header">
        <h1>Product Master Registry</h1>
        <p>Manage central product database, register new products, calibrate batch parameters and generate industrial QR labels.</p>
    </div>
    """, unsafe_allow_html=True)

    p_tabs = st.tabs([
        "📋 Registered Products",
        "➕ Register New Product",
        "✏️ Update Existing Product",
        "🏷️ Generate QR Label"
    ])

    # TAB 1: Registered Products Table
    with p_tabs[0]:
        products = db.get_all_products()
        if products:
            p_df = pd.DataFrame(products)
            p_df["status"] = p_df["expiry_date"].apply(
                lambda exp: "EXPIRED" if (isinstance(exp, (date, datetime)) and exp < date.today()) else "ACTIVE"
            )
            st.dataframe(
                p_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_id": "Product ID",
                    "product_name": "Product Name",
                    "batch_number": "Batch Number",
                    "manufacturing_date": st.column_config.DateColumn("Mfg Date", format="YYYY-MM-DD"),
                    "expiry_date": st.column_config.DateColumn("Expiry Date", format="YYYY-MM-DD"),
                    "status": "Registry Status"
                }
            )
        else:
            st.info("No registered products in database.")

    # TAB 2: Register New Product
    with p_tabs[1]:
        st.markdown("##### Add New Product to Central Registry")
        with st.form("add_product_form", clear_on_submit=True):
            col_np1, col_np2 = st.columns(2)
            with col_np1:
                new_id = st.text_input("Product ID *", placeholder="e.g. PROD002")
                new_name = st.text_input("Product Name *", placeholder="e.g. Organic Aloe Moisturizer")
                new_batch = st.text_input("Batch Number *", placeholder="e.g. BATCH202")
            with col_np2:
                new_mfg = st.date_input("Manufacturing Date *", value=date.today())
                new_exp = st.date_input("Expiry Date *", value=date(date.today().year + 2, date.today().month, date.today().day))

            submit_new = st.form_submit_button("➕ Register Product in MySQL", type="primary")

            if submit_new:
                if not new_id or not new_name or not new_batch:
                    st.error("Validation Error: Product ID, Product Name, and Batch Number are required fields.")
                elif new_mfg > new_exp:
                    st.error("Validation Error: Manufacturing date cannot be after Expiry date.")
                else:
                    success, msg = db.add_product(new_id, new_name, new_batch, new_mfg, new_exp)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # TAB 3: Update Existing Product
    with p_tabs[2]:
        st.markdown("##### Modify Existing Product Record")
        all_prods = db.get_all_products()
        if all_prods:
            prod_ids = [p["product_id"] for p in all_prods]
            sel_pid = st.selectbox("Select Product ID to Edit", prod_ids)
            curr_prod = next((p for p in all_prods if p["product_id"] == sel_pid), None)

            if curr_prod:
                with st.form("update_product_form"):
                    col_u1, col_u2 = st.columns(2)
                    with col_u1:
                        up_name = st.text_input("Product Name", value=curr_prod["product_name"])
                        up_batch = st.text_input("Batch Number", value=curr_prod["batch_number"])
                    with col_u2:
                        up_mfg = st.date_input("Manufacturing Date", value=curr_prod["manufacturing_date"])
                        up_exp = st.date_input("Expiry Date", value=curr_prod["expiry_date"])

                    submit_up = st.form_submit_button("💾 Save Product Updates", type="primary")

                    if submit_up:
                        if up_mfg > up_exp:
                            st.error("Validation Error: Manufacturing date cannot be after Expiry date.")
                        else:
                            success, msg = db.update_product(sel_pid, up_name, up_batch, up_mfg, up_exp)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
        else:
            st.info("No products available to update.")

    # TAB 4: Generate QR Label
    with p_tabs[3]:
        st.markdown("##### Industrial QR Code Generator")
        st.write("Generate and download standardized QR codes for printing onto production labels.")
        qr_prod_id = st.text_input("Target Product ID for QR Encoding", value="PROD001")

        if st.button("Generate QR Code"):
            import qrcode
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(qr_prod_id.strip())
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")

            img_byte_arr = io.BytesIO()
            qr_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            col_qg1, col_qg2 = st.columns([1, 2])
            with col_qg1:
                st.image(img_bytes, caption=f"Encoded: {qr_prod_id}", width=220)
            with col_qg2:
                st.success(f"Standard industrial QR Code generated for '{qr_prod_id}'.")
                st.download_button(
                    label="📥 Download QR Code PNG",
                    data=img_bytes,
                    file_name=f"qr_{qr_prod_id}.png",
                    mime="image/png"
                )


# ==============================================================================
# MODULE 5: ANALYTICS & REPORTS (PHASE 8)
# ==============================================================================
elif menu == "📈 Analytics & Reports":
    st.markdown("""
    <div class="industrial-header">
        <h1>Quality Assurance Analytics & Reporting</h1>
        <p>Defect Pareto analysis, production yield stability metrics, and exportable compliance inspection reports.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = db.get_dashboard_metrics()
    all_history = db.get_inspection_history(limit=500)

    if all_history:
        df_all = pd.DataFrame(all_history)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("##### Failure Mode Pareto Breakdown")
            # Analyze failure causes from remarks
            qr_fails = df_all["qr_status"].value_counts().get("FAIL", 0)
            ocr_fails = df_all["text_status"].value_counts().get("FAIL", 0)
            app_fails = df_all["appearance_status"].value_counts().get("FAIL", 0)

            fail_df = pd.DataFrame([
                {"Stage": "QR Code Failure", "Count": int(qr_fails)},
                {"Stage": "OCR Text Mismatch", "Count": int(ocr_fails)},
                {"Stage": "ORB Appearance Anomaly", "Count": int(app_fails)}
            ])

            pareto_chart = alt.Chart(fail_df).mark_bar(color="#EF4444").encode(
                x=alt.X("Count:Q", title="Defect Occurrences"),
                y=alt.Y("Stage:N", sort="-x", title="Defect Mode"),
                tooltip=["Stage", "Count"]
            ).properties(height=220)
            st.altair_chart(pareto_chart, use_container_width=True)

        with col_a2:
            st.markdown("##### Product-wise Inspection Volume")
            prod_counts = df_all["product_id"].value_counts().reset_index()
            prod_counts.columns = ["Product ID", "Inspections"]

            prod_chart = alt.Chart(prod_counts).mark_bar(color="#38BDF8").encode(
                x=alt.X("Inspections:Q", title="Total Inspections"),
                y=alt.Y("Product ID:N", sort="-x", title="Product Code"),
                tooltip=["Product ID", "Inspections"]
            ).properties(height=220)
            st.altair_chart(prod_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Comprehensive Quality Audit Export")
        csv_full = io.StringIO()
        df_all.to_csv(csv_full, index=False)
        st.download_button(
            label="📥 Download Complete Audit Report (CSV)",
            data=csv_full.getvalue(),
            file_name=f"full_quality_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No inspection records logged in database yet.")


# ==============================================================================
# MODULE 6: SYSTEM & DIAGNOSTICS (PHASE 9 & 10)
# ==============================================================================
elif menu == "⚙️ System & Diagnostics":
    st.markdown("""
    <div class="industrial-header">
        <h1>System Diagnostics & Calibration Matrix</h1>
        <p>Operational health monitoring, OCR engine path verification, database connectivity, and documented threshold parameters.</p>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### Environmental Subsystems")
        conn_ok, ping_val = db.check_connection()
        tess_path = ie.TESSERACT_EXE or "NOT DETECTED"

        st.markdown(f"""
        <table class="diag-table">
            <tr><td class="diag-label">Central MySQL Database</td><td class="diag-val"><span class="status-badge {'badge-pass' if conn_ok else 'badge-fail'}">{'CONNECTED' if conn_ok else 'OFFLINE'}</span> ({ping_val})</td></tr>
            <tr><td class="diag-label">Tesseract OCR Binary</td><td class="diag-val"><span class="status-badge {'badge-pass' if ie.TESSERACT_EXE else 'badge-fail'}">{'FOUND' if ie.TESSERACT_EXE else 'MISSING'}</span></td></tr>
            <tr><td class="diag-label">Tesseract Path</td><td class="diag-val"><small><code>{tess_path}</code></small></td></tr>
            <tr><td class="diag-label">OpenCV Engine Version</td><td class="diag-val"><code>{cv2.__version__}</code></td></tr>
            <tr><td class="diag-label">Streamlit Version</td><td class="diag-val"><code>{st.__version__}</code></td></tr>
        </table>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("##### Documented Inspection Thresholds (Phase 5)")
        st.markdown("""
        <table class="diag-table">
            <tr><td class="diag-label">Mean Brightness</td><td class="diag-val"><code>55.0 - 225.0</code></td></tr>
            <tr><td class="diag-label">Optical Contrast</td><td class="diag-val"><code>&ge; 20.0</code></td></tr>
            <tr><td class="diag-label">Blur Sharpness Index</td><td class="diag-val"><code>&ge; 50.0</code> (Laplacian Var)</td></tr>
            <tr><td class="diag-label">Max Specular Glare</td><td class="diag-val"><code>&lt; 35.0%</code></td></tr>
            <tr><td class="diag-label">ORB Min Keypoint Inliers</td><td class="diag-val"><code>&ge; 15</code> (RANSAC)</td></tr>
            <tr><td class="diag-label">ORB Alignment Ratio</td><td class="diag-val"><code>&ge; 30.0%</code> (Inliers / Matches)</td></tr>
            <tr><td class="diag-label">OCR Text Similarity</td><td class="diag-val"><code>&ge; 85.0%</code> fuzzy match</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Industrial Decision Logic Matrix")
    st.markdown("""
    | Stage | PASS Condition | REVIEW Condition | FAIL Condition (Critical) |
    | :--- | :--- | :--- | :--- |
    | **Lighting & Optics** | Brightness 55-225, Contrast &ge; 20, Sharpness &ge; 50 | Underexposed (&lt;55), Glare &gt;35%, or Blur &lt;50 | Severe optical obstruction |
    | **QR Code** | Valid QR decoded matching registered format | Partially clipped / noisy boundary | QR missing, corrupt, or unreadable |
    | **Database** | Product ID registered and active (not expired) | - | Product ID unknown, or product expired |
    | **OCR Text** | Product name and batch confirmed matching | Uncertain characters, low character count | Confirmed critical text mismatch (tampered label) |
    | **ORB Appearance** | Keypoint inliers &ge; 15, alignment ratio &ge; 30% | Inliers 6 - 14, or reference template missing | Inliers &lt; 6 with adequate features (tampered layout) |
    | **Final Verdict** | **ALL required stages PASS** | **Any stage REVIEW and no critical FAIL** | **ANY critical FAIL strictly overrides all** |
    """)
