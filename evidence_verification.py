"""
Evidence Verification Module for WildAI
Includes:
- Metadata extraction & analysis (EXIF, PNG, XMP, dimensions, timestamps, GPS)
- Interchangeable AI-forensics interface & local ELA / frequency consistency analysis
- Perceptual hashing (pHash) & duplicate detection
- Trivandrum / Kerala boundary & GPS/Reported distance consistency verification
- Camera / CCTV characteristic inspection (resolution, aspect ratio, surveillance flags)
- Evidence Decision Engine with transparent rules and human review outcomes
"""

import os
import math
import hashlib
import json
from datetime import datetime
from PIL import Image, ExifTags
import numpy as np
import cv2

try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

# =====================================================================
# Configuration: Geographic Boundaries for Thiruvananthapuram (Trivandrum)
# =====================================================================
TARGET_REGION = {
    "name": os.getenv("TARGET_REGION_NAME", "Thiruvananthapuram (Trivandrum), Kerala, India"),
    "lat_min": float(os.getenv("TARGET_LAT_MIN", "8.15")),
    "lat_max": float(os.getenv("TARGET_LAT_MAX", "8.95")),
    "lon_min": float(os.getenv("TARGET_LON_MIN", "76.65")),
    "lon_max": float(os.getenv("TARGET_LON_MAX", "77.35")),
    "center_lat": 8.5241,
    "center_lon": 76.9366
}

SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "canva", "pixlr", "affinity", "paint.net",
    "lightroom", "coreldraw", "snapseed"
]

AI_GENERATOR_KEYWORDS = [
    "midjourney", "stable diffusion", "stablediffusion", "dall-e", "dalle",
    "novelai", "deepai", "bing image creator", "firefly", "adobe firefly",
    "flux.1", "flux", "runway", "leonardo.ai", "artbreeder", "comfyui", "automatic1111",
    "generative ai", "ai-generated", "synthetic", "craiyon", "imagine", "wombo",
    "dreamstudio", "civitai", "recraft", "ideogram", "kling", "luma", "sora", "pika"
]

SUPPORTED_MIME_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/webp": [".webp"]
}

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


def calculate_haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    try:
        r = 6371.0  # Earth radius in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(r * c, 2)
    except Exception:
        return None


def convert_exif_ratio_to_float(ratio):
    """Safely converts EXIF ratio / fraction / tuple to float."""
    try:
        if hasattr(ratio, 'num') and hasattr(ratio, 'den'):
            return float(ratio.num) / float(ratio.den) if ratio.den != 0 else 0.0
        if isinstance(ratio, tuple) and len(ratio) == 2:
            return float(ratio[0]) / float(ratio[1]) if ratio[1] != 0 else 0.0
        return float(ratio)
    except Exception:
        return 0.0


def parse_gps_coordinates(tags):
    """
    Parses latitude and longitude from exifread tags dictionary.
    Returns (lat, lon) in decimal degrees or (None, None).
    """
    try:
        lat_tag = tags.get('GPS GPSLatitude')
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_tag = tags.get('GPS GPSLongitude')
        lon_ref = tags.get('GPS GPSLongitudeRef')

        if not lat_tag or not lon_tag or not lat_ref or not lon_ref:
            return None, None

        lat_vals = [convert_exif_ratio_to_float(x) for x in lat_tag.values]
        lon_vals = [convert_exif_ratio_to_float(x) for x in lon_tag.values]

        lat = lat_vals[0] + (lat_vals[1] / 60.0) + (lat_vals[2] / 3600.0)
        lon = lon_vals[0] + (lon_vals[1] / 60.0) + (lon_vals[2] / 3600.0)

        if str(lat_ref).strip().upper() == 'S':
            lat = -lat
        if str(lon_ref).strip().upper() == 'W':
            lon = -lon

        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


# =====================================================================
# 1. Metadata Extraction & Analysis
# =====================================================================
def extract_and_analyze_metadata(file_path):
    """
    Comprehensive server-side metadata extraction and integrity inspection.
    Extracts EXIF, timestamps, GPS, software, dimensions, and determines consistency.
    """
    findings = {
        "status": "success",
        "has_exif": False,
        "capture_timestamp": None,
        "camera_make": None,
        "camera_model": None,
        "lens_model": None,
        "gps_latitude": None,
        "gps_longitude": None,
        "dimensions": None,
        "orientation": None,
        "software_history": [],
        "file_creation_time": None,
        "file_modification_time": None,
        "has_thumbnail": False,
        "format": None,
        "raw_tags_count": 0,
        "metadata_fields_present": [],
        "metadata_fields_missing": [],
        "inconsistencies": [],
        "suspicious_indicators": [],
        "audit_notes": []
    }

    try:
        # File timestamps on server disk (explicitly noted as not proof of capture)
        stat = os.stat(file_path)
        findings["file_creation_time"] = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        findings["file_modification_time"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        findings["audit_notes"].append(
            "Note: File system creation/modification timestamps reflect server write time and are not proof of original capture time."
        )

        findings["ai_generated_indicators"] = []

        with Image.open(file_path) as pil_img:
            findings["dimensions"] = f"{pil_img.width}x{pil_img.height}"
            findings["width"] = pil_img.width
            findings["height"] = pil_img.height
            findings["format"] = pil_img.format
            findings["mode"] = pil_img.mode

            # Check PNG text chunks or XMP
            if pil_img.info:
                for k, v in pil_img.info.items():
                    if isinstance(v, str):
                        v_lower = v.lower()
                        if any(ai_k in v_lower for ai_k in AI_GENERATOR_KEYWORDS):
                            findings["ai_generated_indicators"].append(f"Header '{k}' indicates AI generator: {v}")
                            findings["software_history"].append(f"{k}: {v}")
                        elif any(sw in v_lower for sw in SUSPICIOUS_SOFTWARE):
                            findings["suspicious_indicators"].append(f"Header '{k}' indicates editing software: {v}")
                            findings["software_history"].append(f"{k}: {v}")

        # Extract using ExifRead
        if EXIFREAD_AVAILABLE:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                findings["raw_tags_count"] = len(tags)
                if len(tags) > 0:
                    findings["has_exif"] = True

                # Camera Make / Model
                if 'Image Make' in tags:
                    findings["camera_make"] = str(tags['Image Make']).strip()
                    findings["metadata_fields_present"].append("Camera Make")
                else:
                    findings["metadata_fields_missing"].append("Camera Make")

                if 'Image Model' in tags:
                    findings["camera_model"] = str(tags['Image Model']).strip()
                    findings["metadata_fields_present"].append("Camera Model")
                else:
                    findings["metadata_fields_missing"].append("Camera Model")

                # Lens
                if 'EXIF LensModel' in tags:
                    findings["lens_model"] = str(tags['EXIF LensModel']).strip()
                    findings["metadata_fields_present"].append("Lens Model")

                # Timestamps
                for dt_tag in ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime']:
                    if dt_tag in tags:
                        findings["capture_timestamp"] = str(tags[dt_tag]).strip()
                        findings["metadata_fields_present"].append("Capture Timestamp")
                        break
                if not findings["capture_timestamp"]:
                    findings["metadata_fields_missing"].append("Capture Timestamp")

                # Orientation
                if 'Image Orientation' in tags:
                    findings["orientation"] = str(tags['Image Orientation']).strip()
                    findings["metadata_fields_present"].append("Orientation")

                # Software
                if 'Image Software' in tags:
                    soft_val = str(tags['Image Software']).strip()
                    soft_lower = soft_val.lower()
                    findings["software_history"].append(soft_val)
                    findings["metadata_fields_present"].append("Software Tag")
                    if any(ai_k in soft_lower for ai_k in AI_GENERATOR_KEYWORDS):
                        findings["ai_generated_indicators"].append(f"Image created by generative AI tool: '{soft_val}'")
                    elif any(sw in soft_lower for sw in SUSPICIOUS_SOFTWARE):
                        findings["suspicious_indicators"].append(f"Image edited or processed with: '{soft_val}'")

                # Thumbnail
                if 'JPEGThumbnail' in tags:
                    findings["has_thumbnail"] = True
                    findings["metadata_fields_present"].append("Embedded Thumbnail")

                # GPS Coordinates
                lat, lon = parse_gps_coordinates(tags)
                if lat is not None and lon is not None:
                    findings["gps_latitude"] = lat
                    findings["gps_longitude"] = lon
                    findings["metadata_fields_present"].append("GPS Coordinates")
                else:
                    findings["metadata_fields_missing"].append("GPS Coordinates")
        else:
            findings["audit_notes"].append("ExifRead library unavailable, relied on basic PIL extraction.")

        # Important evidentiary principles
        if not findings["has_exif"]:
            findings["audit_notes"].append(
                "Missing EXIF metadata: Common for CCTV exports, messaging app uploads, or screenshots. Missing metadata does NOT automatically mean the image is fake."
            )
        else:
            findings["audit_notes"].append(
                "EXIF metadata present: Camera model and timestamps can be forged. Metadata presence does not guarantee authenticity."
            )

    except Exception as e:
        findings["status"] = "error"
        findings["error"] = str(e)

    return findings


# =====================================================================
# 2. AI-Generated & Manipulated Image Forensics Interface
# =====================================================================
class ImageForensicsInterface:
    """Interchangeable interface for AI forensics and manipulation evaluation."""

    def evaluate(self, file_path):
        """
        Evaluates image for manipulation or AI generation markers.
        Returns a structured dictionary with transparent findings and limitations.
        """
        external_enabled = os.getenv("ENABLE_EXTERNAL_FORENSICS_MODEL", "false").lower() == "true"
        external_endpoint = os.getenv("FORENSICS_MODEL_ENDPOINT", "").strip()

        if external_enabled and external_endpoint:
            return self._call_external_forensics_service(file_path, external_endpoint)

        # Fallback to local transparent forensic inspection (ELA & frequency analysis)
        return self._local_forensic_inspection(file_path)

    def _call_external_forensics_service(self, file_path, endpoint):
        """Dispatches to configured external forensics microservice/API."""
        import requests
        try:
            api_key = os.getenv("FORENSICS_MODEL_API_KEY", "")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            with open(file_path, "rb") as f:
                resp = requests.post(endpoint, files={"file": f}, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "available": True,
                    "service_type": "external_model",
                    "model_name": data.get("model_name", "ExternalForensicsNet"),
                    "ai_generation_likelihood": data.get("score"),
                    "verdict": data.get("verdict", "INCONCLUSIVE"),
                    "limitations": "External detector score. Single detector outputs are advisory and not definitive proof.",
                    "indicators": data.get("indicators", [])
                }
        except Exception as e:
            return {
                "available": False,
                "service_type": "external_model",
                "error": f"External forensics service request failed: {str(e)}",
                "model_name": "External Service",
                "ai_generation_likelihood": None,
                "verdict": "SERVICE_UNAVAILABLE",
                "limitations": "Service timed out or failed. No fabricated score generated."
            }

        return {
            "available": False,
            "service_type": "none",
            "model_name": None,
            "ai_generation_likelihood": None,
            "verdict": "AI authenticity analysis unavailable",
            "limitations": "No AI forensics model or external service configured."
        }

    def _local_forensic_inspection(self, file_path):
        """
        Local forensic inspection based on Error Level Analysis (ELA) and compression uniformity.
        Does not invent an arbitrary '100% fake' score.
        """
        try:
            img = cv2.imread(file_path)
            if img is None:
                return {
                    "available": False,
                    "verdict": "IMAGE_DECODE_FAILED",
                    "limitations": "Image could not be decoded by OpenCV."
                }

            # Error Level Analysis
            temp_path = f"{file_path}.temp_ela.jpg"
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            compressed = cv2.imread(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if compressed is None:
                return {
                    "available": False,
                    "verdict": "ELA_PROCESSING_FAILED",
                    "limitations": "Failed to create recompressed baseline."
                }

            diff = cv2.absdiff(img, compressed)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mean_diff = float(np.mean(gray_diff))
            std_diff = float(np.std(gray_diff))
            max_diff = float(np.max(gray_diff))

            # Laplacian sharpness check (detect blurry or generated artificial textures)
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray_orig, cv2.CV_64F).var())

            # 2D Fast Fourier Transform (FFT) analysis:
            # AI-generated images typically show anomalous high-frequency grid spectra or unnatural spectral decay
            f = np.fft.fft2(gray_orig)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)
            h_f, w_f = gray_orig.shape
            crow, ccol = h_f // 2, w_f // 2
            # High-frequency band energy
            radius = min(crow, ccol) // 3
            mask = np.ones((h_f, w_f), np.uint8)
            cv2.circle(mask, (ccol, crow), radius, 0, -1)
            high_freq_energy = float(np.mean(magnitude_spectrum[mask == 1]))

            # 3. Bayer CFA demosaicing periodic peak analysis:
            # Physical camera sensors use Bayer color filter arrays (RGGB), creating periodic high-frequency correlation in (R - G) difference.
            # Synthetic AI images lack camera demosaicing artifacts (ratio typically < 40 vs > 80 for real sensor photos).
            b_ch, g_ch, r_ch = cv2.split(img.astype(np.float32))
            d_rg = r_ch - g_ch
            lap_drg = cv2.Laplacian(d_rg, cv2.CV_32F)
            f_demosaic = np.fft.fftshift(np.fft.fft2(lap_drg))
            cfa_ratio = float(np.max(np.abs(f_demosaic)) / (np.mean(np.abs(f_demosaic)) + 1e-7))

            # 4. Patch noise variance on flat areas (sensor shot noise vs synthetic smoothness):
            patch_vars = []
            for r_idx in range(0, h_f - 32, 32):
                for c_idx in range(0, w_f - 32, 32):
                    patch = gray_orig[r_idx:r_idx+32, c_idx:c_idx+32]
                    patch_vars.append(float(np.var(patch)))
            if patch_vars:
                patch_vars.sort()
                flat_patch_var = float(np.mean(patch_vars[:max(1, len(patch_vars)//20)]))
            else:
                flat_patch_var = float(laplacian_var)

            indicators = []
            is_suspected_ai = False

            if mean_diff > 12.0 or std_diff > 16.0:
                indicators.append(f"Non-uniform compression error levels detected (ELA Mean: {mean_diff:.1f}, Std: {std_diff:.1f})")
            if laplacian_var < 15.0:
                indicators.append(f"Low edge variance ({laplacian_var:.1f}): image is unusually blurry or synthetic-smooth")
                is_suspected_ai = True
            elif laplacian_var > 2500.0 and high_freq_energy > 160.0:
                # Unnatural hyper-sharp synthetic textures common in generative upscalers
                indicators.append(f"Hyper-intense high-frequency energy ({high_freq_energy:.1f}): possible diffusion model artifact")
                is_suspected_ai = True

            # Camera sensor physical signature evaluation:
            if cfa_ratio < 30.0 and flat_patch_var < 8.0:
                indicators.append(f"Synthetic texture signature: Absence of optical CFA sensor noise (CFA ratio: {cfa_ratio:.1f}, flat-patch noise: {flat_patch_var:.2f})")
                is_suspected_ai = True

            verdict = "SUSPECTED_AI" if is_suspected_ai else ("POTENTIAL_MANIPULATION_DETECTED" if len(indicators) > 0 else "CONSISTENT_COMPRESSION")

            return {
                "available": True,
                "service_type": "local_ela_heuristics",
                "model_name": "WildAI Physical Sensor & Spectral Forensics v3.0",
                "ela_mean_error": round(mean_diff, 2),
                "ela_std_deviation": round(std_diff, 2),
                "edge_variance": round(laplacian_var, 2),
                "high_freq_energy": round(high_freq_energy, 2),
                "cfa_ratio": round(cfa_ratio, 2),
                "flat_patch_var": round(flat_patch_var, 2),
                "indicators": indicators,
                "is_suspected_ai": is_suspected_ai,
                "ai_generation_likelihood": None,
                "verdict": verdict,
                "limitations": "Combines ELA recompression difference, Bayer CFA sensor analysis, and 2D FFT spectral decay analysis."
            }
        except Exception as e:
            return {
                "available": False,
                "service_type": "local_ela_heuristics",
                "error": str(e),
                "verdict": "ANALYSIS_ERROR",
                "limitations": "Internal processing exception."
            }


# =====================================================================
# 3. Duplicate Detection & Perceptual Hashing
# =====================================================================
def compute_hashes(file_path):
    """Computes SHA-256 cryptographic hash and pHash (perceptual hash)."""
    # SHA-256
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    sha256_hash = sha256.hexdigest()

    # Perceptual hash (pHash)
    phash_str = None
    if IMAGEHASH_AVAILABLE:
        try:
            with Image.open(file_path) as pil_img:
                phash_val = imagehash.phash(pil_img)
                phash_str = str(phash_val)
        except Exception:
            phash_str = None

    return sha256_hash, phash_str


def check_duplicate_evidence(db_session, current_sha256, current_phash, exclude_submission_id=None):
    """
    Checks the database of previously submitted evidence.
    Returns exact duplicates (SHA-256 match) and visual near-duplicates (pHash Hamming distance <= 10).
    A duplicate image is flagged for review, not automatically labeled fraudulent.
    """
    from models import WildlifeEvidenceSubmission
    query = db_session.query(WildlifeEvidenceSubmission)
    if exclude_submission_id:
        query = query.filter(WildlifeEvidenceSubmission.submission_id != exclude_submission_id)

    submissions = query.all()
    exact_matches = []
    near_matches = []

    for sub in submissions:
        # Exact cryptographic hash match
        if sub.sha256_hash and sub.sha256_hash == current_sha256:
            exact_matches.append({
                "submission_id": sub.submission_id,
                "file_name": sub.file_name,
                "timestamp": sub.timestamp.isoformat() if sub.timestamp else None,
                "reported_location": sub.reported_location_name,
                "status": sub.review_status,
                "match_type": "EXACT_FILE_MATCH"
            })
            continue

        # Perceptual hash match
        if IMAGEHASH_AVAILABLE and current_phash and sub.phash:
            try:
                h1 = imagehash.hex_to_hash(current_phash)
                h2 = imagehash.hex_to_hash(sub.phash)
                distance = h1 - h2
                if distance <= 10:  # Perceptually similar threshold
                    confidence = round(max(0.0, 1.0 - (distance / 32.0)), 2)
                    near_matches.append({
                        "submission_id": sub.submission_id,
                        "file_name": sub.file_name,
                        "timestamp": sub.timestamp.isoformat() if sub.timestamp else None,
                        "hamming_distance": int(distance),
                        "similarity_confidence": confidence,
                        "status": sub.review_status,
                        "match_type": "PERCEPTUAL_NEAR_MATCH"
                    })
            except Exception:
                pass

    # Reverse Image Search Provider check
    enable_reverse_search = os.getenv("ENABLE_REVERSE_IMAGE_SEARCH", "false").lower() == "true"
    reverse_search_result = {
        "configured": enable_reverse_search,
        "status": "Online source verification unavailable (no reverse-image-search API configured)" if not enable_reverse_search else "Checked",
        "matches": []
    }

    return {
        "exact_matches": exact_matches,
        "near_matches": near_matches,
        "has_duplicates": len(exact_matches) > 0 or len(near_matches) > 0,
        "reverse_search": reverse_search_result
    }


# =====================================================================
# 4. Location Verification (Trivandrum, Kerala Boundary Engine)
# =====================================================================
def verify_location_evidence(embedded_lat, embedded_lon, reported_lat, reported_lon, reported_text=""):
    """
    Verifies location consistency against target region (Thiruvananthapuram, Kerala, India).
    Classifies into the 5 mandatory evidence outcomes:
    1. GPS consistent with the reported location
    2. Reported location without embedded GPS
    3. GPS inconsistent with the reported location
    4. Location unavailable
    5. Location requires manual review
    """
    result = {
        "target_region": TARGET_REGION["name"],
        "target_boundaries": {
            "lat_min": TARGET_REGION["lat_min"],
            "lat_max": TARGET_REGION["lat_max"],
            "lon_min": TARGET_REGION["lon_min"],
            "lon_max": TARGET_REGION["lon_max"]
        },
        "embedded_gps": {"lat": embedded_lat, "lon": embedded_lon} if embedded_lat and embedded_lon else None,
        "reported_gps": {"lat": reported_lat, "lon": reported_lon} if reported_lat and reported_lon else None,
        "reported_text": reported_text,
        "distance_km": None,
        "embedded_inside_target_region": None,
        "reported_inside_target_region": None,
        "classification": "Location unavailable",
        "explanation": "",
        "flag_for_review": False
    }

    # Check boundaries for embedded GPS
    if embedded_lat is not None and embedded_lon is not None:
        result["embedded_inside_target_region"] = (
            TARGET_REGION["lat_min"] <= embedded_lat <= TARGET_REGION["lat_max"] and
            TARGET_REGION["lon_min"] <= embedded_lon <= TARGET_REGION["lon_max"]
        )

    # Check boundaries for reported GPS
    if reported_lat is not None and reported_lon is not None:
        result["reported_inside_target_region"] = (
            TARGET_REGION["lat_min"] <= reported_lat <= TARGET_REGION["lat_max"] and
            TARGET_REGION["lon_min"] <= reported_lon <= TARGET_REGION["lon_max"]
        )

    # 1. Neither embedded GPS nor reported GPS provided
    if embedded_lat is None and reported_lat is None:
        result["classification"] = "Location unavailable"
        result["explanation"] = "Neither embedded camera GPS metadata nor map coordinates were provided."
        result["flag_for_review"] = True
        return result

    # 2. Reported location provided, but NO embedded camera GPS
    if embedded_lat is None and reported_lat is not None:
        result["classification"] = "Reported location without embedded GPS"
        inside_str = "inside" if result["reported_inside_target_region"] else "outside"
        result["explanation"] = f"User selected map coordinates ({reported_lat:.4f}, {reported_lon:.4f}) {inside_str} {TARGET_REGION['name']}, but image has no embedded camera GPS metadata."
        if not result["reported_inside_target_region"]:
            result["flag_for_review"] = True
            result["explanation"] += " Reported location falls outside the Trivandrum surveillance bounds."
        return result

    # 3. Embedded GPS provided, but NO reported map coordinates
    if embedded_lat is not None and reported_lat is None:
        result["classification"] = "Location requires manual review"
        inside_str = "inside" if result["embedded_inside_target_region"] else "outside"
        result["explanation"] = f"Image contains embedded GPS ({embedded_lat:.4f}, {embedded_lon:.4f}) {inside_str} {TARGET_REGION['name']}, but no reported coordinates were provided by the user."
        if not result["embedded_inside_target_region"]:
            result["flag_for_review"] = True
            result["explanation"] += " GPS points outside Trivandrum boundary."
        return result

    # 4. BOTH embedded GPS and reported GPS are present -> Compare distance
    dist = calculate_haversine_distance_km(embedded_lat, embedded_lon, reported_lat, reported_lon)
    result["distance_km"] = dist

    # Tolerance threshold for distance: 5 km (accounting for forest post vs capture point)
    if dist is not None and dist <= 5.0:
        result["classification"] = "GPS consistent with the reported location"
        result["explanation"] = f"Embedded GPS matches reported location within {dist:.2f} km."
        if not result["embedded_inside_target_region"]:
            result["flag_for_review"] = True
            result["explanation"] += f" Note: Incident location ({embedded_lat:.4f}, {embedded_lon:.4f}) is outside Trivandrum bounds."
    else:
        result["classification"] = "GPS inconsistent with the reported location"
        result["explanation"] = f"Discrepancy detected: Embedded GPS is {dist:.2f} km away from user-selected map coordinates."
        result["flag_for_review"] = True

    return result


# =====================================================================
# 5. Camera & CCTV Characteristics Inspection
# =====================================================================
def inspect_camera_source(metadata_dict, width, height, reported_source="unknown"):
    """
    Evaluates surveillance/camera characteristics (resolution, aspect ratio,
    surveillance typical patterns). Clarifies that missing EXIF is expected for CCTV.
    """
    analysis = {
        "reported_source": reported_source,
        "inferred_characteristics": [],
        "cctv_typical_aspect_ratio": False,
        "is_exif_expected": reported_source not in ["CCTV/IP camera", "trail camera"],
        "notes": []
    }

    aspect_ratio = round(width / height, 2) if height > 0 else 1.0
    analysis["aspect_ratio"] = aspect_ratio
    analysis["resolution"] = f"{width}x{height}"

    # Typical surveillance resolutions (16:9 ~ 1.78, 4:3 ~ 1.33)
    if abs(aspect_ratio - 1.78) < 0.05 or abs(aspect_ratio - 1.33) < 0.05:
        analysis["cctv_typical_aspect_ratio"] = True
        analysis["inferred_characteristics"].append(f"Standard surveillance aspect ratio ({aspect_ratio}:1)")

    if reported_source == "CCTV/IP camera":
        analysis["notes"].append(
            "CCTV Source: Surveillance and IP cameras routinely strip EXIF tags upon video export or DVR frame grab. Absence of EXIF is standard."
        )
    elif reported_source == "trail camera":
        analysis["notes"].append(
            "Trail Camera Source: Some motion-triggered camera traps embed basic EXIF while others imprint timestamp overlays on the frame pixels."
        )
    elif reported_source == "mobile camera" and not metadata_dict.get("has_exif"):
        analysis["notes"].append(
            "Mobile Camera Source without EXIF: Image may have been compressed/shared via messaging applications (WhatsApp/Telegram) which strip EXIF data."
        )

    return analysis


# =====================================================================
# 6. Wildlife Detection Screening (Integrated YOLO11)
# =====================================================================
def screen_wildlife_evidence(cv_image, model):
    """
    Screens image with the existing YOLO model.
    Supported wildlife classes and unsupported requests are clearly indicated.
    """
    result = {
        "model_loaded": model is not None,
        "detections": [],
        "supported_classes": ["elephant", "bear", "zebra", "giraffe", "person"] if model else [],
        "unsupported_classes_note": "Tigers and Wild Boars are not native classes in the standard YOLO-COCO dataset. Detections reflect verifiable model classes only.",
        "highest_confidence": 0.0,
        "primary_detected_species": None
    }

    if model is None or cv_image is None:
        return result

    try:
        yolo_res = model.predict(cv_image, conf=0.30, verbose=False)
        highest_c = 0.0
        primary_sp = None

        if len(yolo_res) > 0 and yolo_res[0].boxes:
            for box in yolo_res[0].boxes:
                c = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = model.names.get(cls_idx, f"class_{cls_idx}")
                xyxy = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]

                result["detections"].append({
                    "class": class_name,
                    "confidence": round(c, 3),
                    "bbox": xyxy
                })

                if c > highest_c:
                    highest_c = c
                    primary_sp = class_name

        result["highest_confidence"] = round(highest_c, 3)
        result["primary_detected_species"] = primary_sp
    except Exception as e:
        result["error"] = str(e)

    return result


# =====================================================================
# 7. Evidence Decision Engine
# =====================================================================
def evaluate_evidence_decision(
    file_valid,
    metadata_res,
    forensics_res,
    duplicate_res,
    location_res,
    cctv_res,
    user_data
):
    """
    Transparent rule-based decision engine.
    Produces one of four outcomes:
    - ACCEPTED FOR REVIEW (no major issues)
    - FLAGGED FOR REVIEW (conflicting, incomplete, or suspicious evidence)
    - REJECTED (invalid file, corrupted image, or severe policy violation)
    - ANALYSIS INCOMPLETE (required service failed)
    """
    flags = []
    explanations = []

    # 1. Technical File Validation Check
    if not file_valid.get("valid"):
        return {
            "outcome": "REJECTED — INVALID FILE",
            "explanation": f"File rejected: {file_valid.get('reason', 'Invalid image file or decoding failed')}",
            "flags": ["FILE_VALIDATION_FAILURE"]
        }

    # 2. Check for AI-Generated & Synthetic Manipulation Indicators
    if metadata_res.get("ai_generated_indicators"):
        ai_details = "; ".join(metadata_res["ai_generated_indicators"])
        flags.append("SUSPECTED_AI")
        explanations.append(f"Suspected AI-generated image ({ai_details}). Flagged for administrative verification.")

    if forensics_res.get("verdict") in ["SUSPECTED_AI", "AI_GENERATED"] or forensics_res.get("is_suspected_ai"):
        flags.append("SUSPECTED_AI")
        explanations.append("Forensic spectral / ELA inspection indicates potential synthetic AI generation. Flagged for officer review.")

    # 3. Check for Service Failures
    if forensics_res.get("verdict") == "SERVICE_UNAVAILABLE" and forensics_res.get("service_type") == "external_model":
        return {
            "outcome": "ANALYSIS INCOMPLETE",
            "explanation": "External forensics service was required but unavailable. Submission queued for retry.",
            "flags": ["FORENSICS_SERVICE_UNAVAILABLE"]
        }

    # 3. Check Duplicate Evidence
    if duplicate_res.get("exact_matches"):
        flags.append("EXACT_DUPLICATE_FOUND")
        explanations.append(f"Exact cryptographic match with {len(duplicate_res['exact_matches'])} prior submission(s). Flagged for reviewer inspection.")
    elif duplicate_res.get("near_matches"):
        flags.append("NEAR_DUPLICATE_FOUND")
        explanations.append(f"Visually similar to {len(duplicate_res['near_matches'])} prior submission(s).")

    # 4. Check Metadata & Forensics
    if metadata_res.get("suspicious_indicators"):
        flags.append("SUSPICIOUS_SOFTWARE_HISTORY")
        for ind in metadata_res["suspicious_indicators"]:
            explanations.append(ind)

    if forensics_res.get("indicators"):
        flags.append("POTENTIAL_IMAGE_MANIPULATION")
        for ind in forensics_res["indicators"]:
            explanations.append(ind)

    # 5. Check Location Consistency
    loc_classification = location_res.get("classification")
    if loc_classification == "GPS inconsistent with the reported location":
        flags.append("GPS_REPORTED_LOCATION_MISMATCH")
        explanations.append(location_res.get("explanation", "Discrepancy between GPS metadata and user location."))
    elif loc_classification == "Location unavailable":
        flags.append("NO_LOCATION_COORDINATES")
        explanations.append("No GPS coordinates available from camera or map selection.")

    if location_res.get("embedded_inside_target_region") is False:
        flags.append("GPS_OUTSIDE_TARGET_REGION")
        explanations.append("GPS coordinates indicate capture outside Thiruvananthapuram regional boundary (flagged for review, not rejected).")

    # Determine Final Outcome adhering to mandatory standards:
    # 1. AI-GENERATED — NOT AUTHENTIC
    if "SUSPECTED_AI" in flags:
        outcome = "AI-GENERATED — NOT AUTHENTIC"
        final_explanation = "AI-generated image detected. Upload rejected."
        return {
            "outcome": outcome,
            "explanation": final_explanation,
            "flags": flags
        }

    # 2. MANIPULATED — NOT AUTHENTIC
    if "POTENTIAL_IMAGE_MANIPULATION" in flags or "SUSPICIOUS_SOFTWARE_HISTORY" in flags:
        outcome = "MANIPULATED — NOT AUTHENTIC"
        final_explanation = "Image manipulation detected. Upload rejected."
        return {
            "outcome": outcome,
            "explanation": final_explanation,
            "flags": flags
        }

    # 3. Check for Service Failure: VERIFICATION INCOMPLETE
    if "FORENSICS_SERVICE_UNAVAILABLE" in flags:
        return {
            "outcome": "VERIFICATION INCOMPLETE",
            "explanation": "Image verification incomplete. Required service was unavailable.",
            "flags": flags
        }

    # 4. CAMERA/CCTV EVIDENCE — VERIFIED vs UNVERIFIED — MANUAL REVIEW REQUIRED
    # Real camera/CCTV photos with valid capture evidence:
    has_optical_camera = metadata_res.get("has_exif") and bool(metadata_res.get("camera_make") or metadata_res.get("camera_model"))
    is_cctv_surveillance = cctv_res.get("cctv_typical_aspect_ratio") or user_data.get("source_type") in ["CCTV/IP camera", "trail camera"]

    if len(flags) == 0:
        if has_optical_camera or is_cctv_surveillance:
            outcome = "CAMERA/CCTV EVIDENCE — VERIFIED"
            final_explanation = "Camera/CCTV evidence passed verification. Awaiting required approval."
        else:
            outcome = "UNVERIFIED — MANUAL REVIEW REQUIRED"
            final_explanation = "Image verification inconclusive. Manual review required."
    else:
        outcome = "UNVERIFIED — MANUAL REVIEW REQUIRED"
        final_explanation = "Verification flags identified: " + "; ".join(explanations)

    return {
        "outcome": outcome,
        "explanation": final_explanation,
        "flags": flags
    }
