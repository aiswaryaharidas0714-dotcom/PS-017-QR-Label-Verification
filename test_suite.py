import os
import sys
import unittest
from datetime import datetime, date, timedelta
from pathlib import Path
import cv2
import numpy as np

import database as db
import inspection_engine as ie


class IndustrialInspectionTestSuite(unittest.TestCase):
    """
    Comprehensive verification test suite covering all 12 operational scenarios
    required by Sakthi Hackathon 2.0 PS-017 specification (Phase 10).
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "="*80)
        print("PS-017 INDUSTRIAL AUTOMATED INSPECTION TEST SUITE")
        print("="*80)
        # Ensure sample images exist
        cls.sample_path = "sample_product_label.png"
        if not Path(cls.sample_path).exists():
            cls.sample_path = "captured_images/sample_product_label.png"
        cls.sample_img = cv2.imread(cls.sample_path)
        assert cls.sample_img is not None, f"Sample label not found at {cls.sample_path}"

        cls.altered_path = "altered_product_label.png"
        if not Path(cls.altered_path).exists():
            cls.altered_path = "captured_images/fake_product_label.png"
        cls.altered_img = cv2.imread(cls.altered_path)
        assert cls.altered_img is not None, f"Altered label not found at {cls.altered_path}"

    # TEST 1: Valid product label (End-to-End PASS)
    def test_01_valid_product_label(self):
        print("\n[TEST 1] Testing Valid Product Label (sample_product_label.png)...")
        light = ie.analyze_lighting(self.sample_img)
        self.assertEqual(light["quality_verdict"], "PASS")

        qr_st, qr_val, pts, anno, qr_msg = ie.detect_qr(self.sample_img)
        self.assertEqual(qr_st, "PASS")
        self.assertEqual(qr_val, "PROD001")

        db_st, prod, db_msg = ie.validate_product_in_database(qr_val)
        self.assertEqual(db_st, "PASS")
        self.assertIsNotNone(prod)

        ocr_st, ocr_txt, fields, ocr_msg = ie.verify_ocr_text(self.sample_img, prod)
        self.assertEqual(ocr_st, "PASS")

        app_st, app_met, vis, app_msg = ie.verify_appearance_orb(self.sample_img)
        self.assertEqual(app_st, "PASS")
        self.assertGreaterEqual(app_met["inliers"], 15)

        verdict, summary = ie.evaluate_inspection(
            light,
            {"status": qr_st, "message": qr_msg},
            {"status": db_st, "message": db_msg},
            {"status": ocr_st, "message": ocr_msg},
            {"status": app_st, "metrics": app_met, "message": app_msg}
        )
        self.assertEqual(verdict, "PASS")
        print(f" -> Outcome: {verdict} | Summary: {summary[:90]}...")

    # TEST 2: Unknown product ID
    def test_02_unknown_product(self):
        print("\n[TEST 2] Testing Unknown Product ID ('PROD_UNKNOWN_999')...")
        db_st, prod, db_msg = ie.validate_product_in_database("PROD_UNKNOWN_999")
        self.assertEqual(db_st, "FAIL")
        self.assertIsNone(prod)
        self.assertIn("UNKNOWN PRODUCT", db_msg)

        # Verify critical failure override
        light = {"quality_verdict": "PASS", "message": "Optimal"}
        verdict, summary = ie.evaluate_inspection(
            light,
            {"status": "PASS", "message": "QR Valid"},
            {"status": db_st, "message": db_msg},
            {"status": "PASS", "message": "OCR Valid"},
            {"status": "PASS", "message": "Appearance Valid"}
        )
        self.assertEqual(verdict, "FAIL")
        print(f" -> Outcome: {verdict} | DB Status: {db_st} | Reason: {db_msg}")

    # TEST 3: Missing QR Code
    def test_03_missing_qr_code(self):
        print("\n[TEST 3] Testing Missing / Unreadable QR Code...")
        # Image without QR (crop top text portion only)
        no_qr_img = self.sample_img[0:300, 0:500].copy()
        qr_st, qr_val, pts, anno, qr_msg = ie.detect_qr(no_qr_img)
        self.assertEqual(qr_st, "FAIL")
        self.assertIsNone(qr_val)

        light = {"quality_verdict": "PASS", "message": "Optimal"}
        verdict, summary = ie.evaluate_inspection(
            light,
            {"status": qr_st, "message": qr_msg},
            {"status": "FAIL", "message": "Skipped"},
            {"status": "FAIL", "message": "Skipped"},
            {"status": "PASS", "message": "Appearance Valid"}
        )
        self.assertEqual(verdict, "FAIL")
        print(f" -> Outcome: {verdict} | QR Status: {qr_st} | Reason: {qr_msg}")

    # TEST 4: Incorrect / Counterfeit Product Name
    def test_04_incorrect_product_name(self):
        print("\n[TEST 4] Testing Tampered Label Name ('FAKE PRODUCTLABEL')...")
        prod = {
            "product_id": "PROD001",
            "product_name": "Herbal Shampoo",
            "batch_number": "BATCH101",
            "manufacturing_date": date(2026, 11, 1),
            "expiry_date": date(2028, 11, 1)
        }
        ocr_st, ocr_txt, fields, ocr_msg = ie.verify_ocr_text(self.altered_img, prod)
        self.assertEqual(ocr_st, "FAIL")
        self.assertIn("CRITICAL TEXT MISMATCH", ocr_msg)

        light = {"quality_verdict": "PASS", "message": "Optimal"}
        verdict, summary = ie.evaluate_inspection(
            light,
            {"status": "PASS", "message": "QR Valid"},
            {"status": "PASS", "message": "DB Valid"},
            {"status": ocr_st, "message": ocr_msg},
            {"status": "PASS", "message": "Appearance Valid"}
        )
        self.assertEqual(verdict, "FAIL")
        print(f" -> Outcome: {verdict} | OCR Status: {ocr_st} | Reason: {ocr_msg}")

    # TEST 5: Expired Product
    def test_05_expired_product(self):
        print("\n[TEST 5] Testing Expired Product Detection...")
        # Create expired product simulation
        expired_date = date.today() - timedelta(days=45)
        expired_prod_id = "PROD_EXP_TEST"

        product_dict = {
            "product_id": expired_prod_id,
            "product_name": "Test Expired Syrup",
            "batch_number": "BATCH_OLD",
            "manufacturing_date": date(2020, 1, 1),
            "expiry_date": expired_date
        }

        # Mock query return in inspection_engine
        orig_get_product = ie.get_product
        try:
            ie.get_product = lambda pid: product_dict if pid == expired_prod_id else orig_get_product(pid)
            db_st, prod, db_msg = ie.validate_product_in_database(expired_prod_id)
            self.assertEqual(db_st, "FAIL")
            self.assertIn("EXPIRED", db_msg)

            light = {"quality_verdict": "PASS", "message": "Optimal"}
            verdict, summary = ie.evaluate_inspection(
                light,
                {"status": "PASS", "message": "QR Valid"},
                {"status": db_st, "message": db_msg},
                {"status": "PASS", "message": "OCR Valid"},
                {"status": "PASS", "message": "Appearance Valid"}
            )
            self.assertEqual(verdict, "FAIL")
            print(f" -> Outcome: {verdict} | DB Status: {db_st} | Reason: {db_msg}")
        finally:
            ie.get_product = orig_get_product

    # TEST 6: Blurred / Out-of-focus Image
    def test_06_blurred_image(self):
        print("\n[TEST 6] Testing Blurred / Out-of-Focus Image...")
        blurred_img = cv2.GaussianBlur(self.sample_img, (35, 35), 0)
        light = ie.analyze_lighting(blurred_img)
        self.assertEqual(light["quality_verdict"], "REVIEW")
        self.assertEqual(light["status"], "BLURRED")

        verdict, summary = ie.evaluate_inspection(
            light,
            {"status": "PASS", "message": "QR Valid"},
            {"status": "PASS", "message": "DB Valid"},
            {"status": "PASS", "message": "OCR Valid"},
            {"status": "PASS", "message": "Appearance Valid"}
        )
        self.assertEqual(verdict, "REVIEW")
        print(f" -> Outcome: {verdict} | Quality Status: {light['status']} | Sharpness: {light['sharpness']}")

    # TEST 7: Poor Lighting (Underexposed & Excessive Glare)
    def test_07_poor_lighting(self):
        print("\n[TEST 7] Testing Poor Lighting (Underexposure & Glare)...")
        # 7A: Underexposure (Dark)
        dark_img = np.clip(self.sample_img.astype(np.float32) * 0.15, 0, 255).astype(np.uint8)
        light_dark = ie.analyze_lighting(dark_img)
        self.assertEqual(light_dark["quality_verdict"], "REVIEW")
        self.assertEqual(light_dark["status"], "POOR")
        print(f" -> Dark Image Outcome: {light_dark['status']} (Brightness: {light_dark['brightness']})")

        # 7B: Washed-out Glare
        glare_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        # Add slight faint patch so it's not totally 0 contrast
        glare_img[100:150, 100:150] = 240
        light_glare = ie.analyze_lighting(glare_img)
        self.assertEqual(light_glare["quality_verdict"], "REVIEW")
        print(f" -> Glare Image Outcome: {light_glare['status']} (Glare: {light_glare['glare_pct']}%, Contrast: {light_glare['contrast']})")

    # TEST 8: Appearance Mismatch (ORB Tampering / Alien Image)
    def test_08_appearance_mismatch(self):
        print("\n[TEST 8] Testing Appearance Mismatch with Alien Pattern...")
        # Create an image full of high-frequency texture (lots of keypoints, zero reference matches)
        alien_img = np.zeros((650, 1000, 3), dtype=np.uint8)
        for i in range(0, 1000, 20):
            cv2.line(alien_img, (i, 0), (1000 - i, 650), (255, 255, 255), 2)
        cv2.putText(alien_img, "UNRELATED MACHINERY WARNING", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        app_st, app_met, vis, app_msg = ie.verify_appearance_orb(alien_img)
        self.assertEqual(app_st, "FAIL")
        self.assertLess(app_met["inliers"], 6)

        light = {"quality_verdict": "PASS", "message": "Optimal"}
        verdict, summary = ie.evaluate_inspection(
            light,
            {"status": "PASS", "message": "QR Valid"},
            {"status": "PASS", "message": "DB Valid"},
            {"status": "PASS", "message": "OCR Valid"},
            {"status": app_st, "metrics": app_met, "message": app_msg}
        )
        self.assertEqual(verdict, "FAIL")
        print(f" -> Outcome: {verdict} | Appearance Status: {app_st} | Inliers: {app_met['inliers']} | Reason: {app_msg}")

    # TEST 9: Database Unavailable Handling
    def test_09_database_unavailable_handling(self):
        print("\n[TEST 9] Testing Database Unavailable / Fault Injection Handling...")
        # Temporarily mock get_db_credentials to return invalid config
        orig_creds = db.get_db_credentials
        try:
            db.get_db_credentials = lambda: {
                "host": "invalid-nonexistent-host.local",
                "port": 9999,
                "user": "fake_user",
                "password": "wrong_password",
                "database": "none"
            }
            conn = db.connect_database()
            self.assertIsNone(conn, "Expected connect_database() to return None on failure.")

            # Test safe query handling
            prod = db.get_product("PROD001")
            self.assertIsNone(prod)

            # Test safe save handling
            saved_ok, saved_err = db.save_inspection("TEST", "PASS", "PASS", "PASS", "PASS", "Test")
            self.assertFalse(saved_ok)
            print(" -> Outcome: Handled gracefully without unhandled exceptions. Error message captured.")
        finally:
            db.get_db_credentials = orig_creds

    # TEST 10: Saving Inspection History (Parameterized SQL)
    def test_10_save_inspection_history(self):
        print("\n[TEST 10] Testing Saving Inspection History Record...")
        test_remarks = f"Automated Unit Test Record - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ok, result = db.save_inspection(
            product_id="PROD001",
            qr_status="PASS",
            text_status="PASS",
            appearance_status="PASS",
            final_result="PASS",
            remarks=test_remarks
        )
        self.assertTrue(ok, f"Failed to save inspection: {result}")
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        print(f" -> Outcome: SUCCESS | Record created with Inspection ID: #{result}")

    # TEST 11: Retrieving Inspection History
    def test_11_retrieve_inspection_history(self):
        print("\n[TEST 11] Testing Retrieving Inspection History from MySQL...")
        history = db.get_inspection_history(limit=5)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        first = history[0]
        required_keys = [
            "inspection_id", "product_id", "inspection_date",
            "qr_status", "text_status", "appearance_status", "final_result", "remarks"
        ]
        for k in required_keys:
            self.assertIn(k, first)
        print(f" -> Outcome: SUCCESS | Retrieved {len(history)} recent records with all schema columns intact.")

    # TEST 12: Dashboard Statistics Computation from Real MySQL Records
    def test_12_dashboard_statistics_real_mysql(self):
        print("\n[TEST 12] Testing Dashboard Statistics from Live MySQL Records...")
        metrics = db.get_dashboard_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total", metrics)
        self.assertIn("pass", metrics)
        self.assertIn("review", metrics)
        self.assertIn("fail", metrics)
        self.assertIn("pass_rate", metrics)

        # Check arithmetic consistency: total == pass + review + fail
        calculated_total = metrics["pass"] + metrics["review"] + metrics["fail"]
        self.assertEqual(metrics["total"], calculated_total)

        if metrics["total"] > 0:
            expected_rate = round((metrics["pass"] / metrics["total"]) * 100, 1)
            self.assertAlmostEqual(metrics["pass_rate"], expected_rate, delta=0.2)

        print(f" -> Outcome: SUCCESS | Real Metrics: Total={metrics['total']}, PASS={metrics['pass']}, REVIEW={metrics['review']}, FAIL={metrics['fail']}, Pass Rate={metrics['pass_rate']}%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
