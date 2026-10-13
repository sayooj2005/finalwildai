"""
Comprehensive Acceptance Criteria Test Suite for
WILDAI Image Authenticity and Wildlife Evidence Verification Module

Tests:
1. Genuine mobile-camera image with EXIF (Make, Model, Timestamp).
2. Genuine CCTV screenshot without EXIF (ensuring missing metadata does NOT cause rejection).
3. Image with GPS inside the target region (Trivandrum).
4. Image with GPS outside the target region (Flagged for review, not rejected).
5. Downloaded/forwarded image with missing metadata.
6. Image with suspicious editing metadata (Photoshop/GIMP).
7. Simulated AI-generated / composite test image (ELA flags).
8. Duplicate submission (Perceptual & Cryptographic hash duplicate detection).
9. Corrupted or unsupported file (Technical file rejection).
10. Unavailable external verification service (Reports unavailable without fabricating scores).
11. Verification Decision Engine outcome accuracy.
"""

import os
import io
import json
import unittest
import numpy as np
from PIL import Image, ExifTags
from database import engine, SessionLocal
import models
import evidence_verification

class TestEvidenceVerificationModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        models.Base.metadata.create_all(bind=engine)
        cls.test_dir = os.path.join("temp", "test_evidence")
        os.makedirs(cls.test_dir, exist_ok=True)

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def create_dummy_image(self, filename, width=640, height=480, color=(100, 150, 200), exif_data=None):
        """Helper to create dummy test images with optional EXIF tags."""
        img = Image.new("RGB", (width, height), color=color)
        filepath = os.path.join(self.test_dir, filename)

        if exif_data:
            exif = img.getexif()
            for tag_id, val in exif_data.items():
                exif[tag_id] = val
            img.save(filepath, "JPEG", exif=exif)
        else:
            img.save(filepath, "JPEG")
        return filepath

    # 1. Genuine mobile-camera image with EXIF
    def test_01_genuine_mobile_camera_with_exif(self):
        # 0x010f = Make, 0x0110 = Model, 0x0132 = DateTime
        exif_dict = {
            0x010f: "Apple",
            0x0110: "iPhone 13",
            0x0132: "2026:04:10 14:22:15"
        }
        img_path = self.create_dummy_image("mobile_camera.jpg", exif_data=exif_dict)
        meta = evidence_verification.extract_and_analyze_metadata(img_path)
        
        self.assertTrue(meta["has_exif"])
        self.assertEqual(meta["camera_make"], "Apple")
        self.assertEqual(meta["camera_model"], "iPhone 13")
        self.assertIn("Capture Timestamp", meta["metadata_fields_present"])

    # 2. Genuine CCTV screenshot without EXIF
    def test_02_cctv_without_exif_not_rejected(self):
        img_path = self.create_dummy_image("cctv_frame.jpg", width=1920, height=1080)
        meta = evidence_verification.extract_and_analyze_metadata(img_path)
        cctv_info = evidence_verification.inspect_camera_source(meta, 1920, 1080, reported_source="CCTV/IP camera")
        
        self.assertFalse(meta["has_exif"])
        self.assertTrue(cctv_info["cctv_typical_aspect_ratio"])
        
        # Test Decision Engine does NOT reject missing EXIF for CCTV
        decision = evidence_verification.evaluate_evidence_decision(
            file_valid={"valid": True},
            metadata_res=meta,
            forensics_res={"available": True, "verdict": "CONSISTENT_COMPRESSION", "indicators": []},
            duplicate_res={"exact_matches": [], "near_matches": []},
            location_res={"classification": "Reported location without embedded GPS", "embedded_inside_target_region": None},
            cctv_res=cctv_info,
            user_data={"source_type": "CCTV/IP camera"}
        )
        self.assertNotEqual(decision["outcome"], "REJECTED")

    # 3. Image with GPS inside the target region (Trivandrum)
    def test_03_gps_inside_trivandrum(self):
        ponmudi_lat = 8.7599
        ponmudi_lon = 77.1169
        loc = evidence_verification.verify_location_evidence(
            embedded_lat=ponmudi_lat,
            embedded_lon=ponmudi_lon,
            reported_lat=8.7590,
            reported_lon=77.1160,
            reported_text="Ponmudi Forest Post"
        )
        self.assertEqual(loc["classification"], "GPS consistent with the reported location")
        self.assertTrue(loc["embedded_inside_target_region"])
        self.assertLess(loc["distance_km"], 2.0)

    # 4. Image with GPS outside the target region
    def test_04_gps_outside_target_region_flagged_not_rejected(self):
        delhi_lat = 28.6139
        delhi_lon = 77.2090
        loc = evidence_verification.verify_location_evidence(
            embedded_lat=delhi_lat,
            embedded_lon=delhi_lon,
            reported_lat=delhi_lat,
            reported_lon=delhi_lon
        )
        self.assertFalse(loc["embedded_inside_target_region"])
        
        # Check Decision Engine flags for review, does NOT reject
        decision = evidence_verification.evaluate_evidence_decision(
            file_valid={"valid": True},
            metadata_res={"suspicious_indicators": []},
            forensics_res={"verdict": "CONSISTENT_COMPRESSION", "indicators": []},
            duplicate_res={"exact_matches": [], "near_matches": []},
            location_res=loc,
            cctv_res={},
            user_data={}
        )
        self.assertEqual(decision["outcome"], "UNVERIFIED — MANUAL REVIEW REQUIRED")
        self.assertIn("GPS_OUTSIDE_TARGET_REGION", decision["flags"])

    # 5. Downloaded image with missing metadata
    def test_05_downloaded_image_missing_metadata(self):
        img_path = self.create_dummy_image("downloaded.jpg")
        meta = evidence_verification.extract_and_analyze_metadata(img_path)
        self.assertFalse(meta["has_exif"])
        self.assertIn("Camera Model", meta["metadata_fields_missing"])

    # 6. Image with suspicious editing metadata
    def test_06_suspicious_editing_metadata(self):
        exif_dict = {
            0x0131: "Adobe Photoshop 2024" # 0x0131 = Software
        }
        img_path = self.create_dummy_image("edited_photo.jpg", exif_data=exif_dict)
        meta = evidence_verification.extract_and_analyze_metadata(img_path)
        self.assertTrue(len(meta["suspicious_indicators"]) > 0)
        self.assertIn("Adobe Photoshop", meta["software_history"][0])

    # 7. AI-forensics interchangeable interface
    def test_07_forensics_interface_transparency(self):
        img_path = self.create_dummy_image("forensics_test.jpg")
        forensics = evidence_verification.ImageForensicsInterface()
        res = forensics.evaluate(img_path)
        self.assertTrue(res["available"])
        self.assertIn("limitations", res)
        # Should not invent certainty percentage
        self.assertIsNone(res.get("ai_generation_likelihood"))

    # 8. Duplicate detection using pHash & SHA-256
    def test_08_duplicate_detection(self):
        import uuid
        img_path = self.create_dummy_image("dup_test_1.jpg")
        sha, phash = evidence_verification.compute_hashes(img_path)
        
        # Save a record into test db with unique id
        test_id = f"EV-TEST-DUP-{uuid.uuid4().hex[:6]}"
        sub = models.WildlifeEvidenceSubmission(
            submission_id=test_id,
            file_name="dup_test_1.jpg",
            file_path=img_path,
            sha256_hash=sha,
            phash=phash,
            decision_outcome="CAMERA/CCTV EVIDENCE — VERIFIED",
            review_status="PENDING"
        )
        self.db.add(sub)
        self.db.commit()

        # Check duplicate
        dup_check = evidence_verification.check_duplicate_evidence(
            db_session=self.db,
            current_sha256=sha,
            current_phash=phash,
            exclude_submission_id="EV-TEST-DUP-NEW"
        )
        self.assertTrue(dup_check["has_duplicates"])
        self.assertGreaterEqual(len(dup_check["exact_matches"]), 1)

    # 9. Corrupted or unsupported file rejection
    def test_09_corrupted_file_rejection(self):
        corrupted_path = os.path.join(self.test_dir, "corrupted.jpg")
        with open(corrupted_path, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES")
            
        decision = evidence_verification.evaluate_evidence_decision(
            file_valid={"valid": False, "reason": "Corrupted or non-decodable image file."},
            metadata_res={},
            forensics_res={},
            duplicate_res={},
            location_res={},
            cctv_res={},
            user_data={}
        )
        self.assertEqual(decision["outcome"], "REJECTED — INVALID FILE")

    # 10. Unavailable external verification service does not fabricate scores
    def test_10_unavailable_service_no_fabrication(self):
        os.environ["ENABLE_EXTERNAL_FORENSICS_MODEL"] = "true"
        os.environ["FORENSICS_MODEL_ENDPOINT"] = "http://127.0.0.1:9999/unreachable_detector"
        
        forensics = evidence_verification.ImageForensicsInterface()
        res = forensics.evaluate(self.create_dummy_image("service_down.jpg"))
        
        self.assertFalse(res["available"])
        self.assertEqual(res["verdict"], "SERVICE_UNAVAILABLE")
        self.assertIsNone(res["ai_generation_likelihood"])
        
        # Reset environment variable
        os.environ["ENABLE_EXTERNAL_FORENSICS_MODEL"] = "false"

    # 11. AI-generated image must be AI-GENERATED — NOT AUTHENTIC
    def test_11_ai_generated_image_flagged_for_review(self):
        exif_dict = {
            0x0131: "Midjourney v6.0"  # 0x0131 = Software
        }
        img_path = self.create_dummy_image("ai_generated_midjourney.jpg", exif_data=exif_dict)
        meta = evidence_verification.extract_and_analyze_metadata(img_path)
        self.assertTrue(len(meta["ai_generated_indicators"]) > 0)
        
        decision = evidence_verification.evaluate_evidence_decision(
            file_valid={"valid": True},
            metadata_res=meta,
            forensics_res={"verdict": "CONSISTENT_COMPRESSION", "indicators": []},
            duplicate_res={"exact_matches": [], "near_matches": []},
            location_res={"classification": "Reported location without embedded GPS"},
            cctv_res={},
            user_data={}
        )
        self.assertEqual(decision["outcome"], "AI-GENERATED — NOT AUTHENTIC")
        self.assertIn("SUSPECTED_AI", decision["flags"])

if __name__ == "__main__":
    unittest.main()
