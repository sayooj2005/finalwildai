from sqlalchemy import Boolean, Column, Integer, String, DateTime, Float
from datetime import datetime
from database import Base

class DetectionHistory(Base):
    __tablename__ = "detection_history"
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String, index=True)
    confidence = Column(String)
    danger_level = Column(String)
    image_path = Column(String)
    location = Column(String)
    behaviour = Column(String, default="Walking / Moving")
    pose_pattern = Column(String, default="Standard Posture")
    risk_score = Column(Integer, default=50)
    group_size = Column(Integer, default=1)
    sector_zone = Column(String, default="Zone 1 - Forest Border")
    movement_vector = Column(String, default="Approaching Settlement")
    environmental_factors = Column(String, default="Clear Day, Normal Visibility")
    timestamp = Column(DateTime, default=datetime.now)

class IntrusionAlert(Base):
    __tablename__ = "intrusion_alerts"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)
    location = Column(String)
    status = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

class EventAlert(Base):
    __tablename__ = "event_alerts"
    id = Column(Integer, primary_key=True, index=True)
    alert_level = Column(String, default="INFO")  # INFO, WARNING, HIGH, CRITICAL
    timestamp = Column(DateTime, default=datetime.now)
    camera_id = Column(String)
    location = Column(String)
    species = Column(String)
    track_id = Column(String, index=True)
    risk_level = Column(String)
    confidence = Column(Float)
    reason = Column(String)
    snapshot_path = Column(String, nullable=True)
    video_path = Column(String, nullable=True)
    acknowledgement_status = Column(String, default="PENDING")  # Legacy
    lifecycle_status = Column(String, default="DETECTED") # DETECTED -> CONFIRMED -> ALERTED -> ACKNOWLEDGED -> RESPONDING -> RESOLVED
    assigned_officer = Column(String, nullable=True)
    is_simulated = Column(Boolean, default=False)

class CommunityReport(Base):
    __tablename__ = "community_reports"
    id = Column(Integer, primary_key=True, index=True)
    location_lat = Column(String)
    location_lon = Column(String)
    location_text = Column(String)
    date = Column(String)
    time = Column(String)
    species = Column(String)
    notes = Column(String)
    contact = Column(String)
    is_authentic = Column(Boolean, default=False)
    image_path = Column(String)
    status = Column(String, default="pending")  # pending, verified, rejected
    timestamp = Column(DateTime, default=datetime.now)

class CommunityReportV2(Base):
    __tablename__ = "community_reports_v2"
    id = Column(Integer, primary_key=True, index=True)
    location_text = Column(String)
    location_lat = Column(String, nullable=True)
    location_lon = Column(String, nullable=True)
    description = Column(String, nullable=True)
    image_path = Column(String)
    
    # AI Pipeline Results
    authenticity_result = Column(String)  # likely genuine, potentially manipulated, inconclusive
    authenticity_confidence = Column(Float)
    ai_species = Column(String, nullable=True)
    ai_confidence = Column(Float, nullable=True)
    exif_data = Column(String, nullable=True)  # JSON string of metadata
    
    # Review System
    status = Column(String, default="PENDING")  # PENDING, UNDER REVIEW, VERIFIED, REJECTED
    timestamp = Column(DateTime, default=datetime.now)
    reporter_ip_hash = Column(String)  # Privacy-preserving abuse prevention

class EvacuationRoute(Base):
    __tablename__ = "evacuation_routes"
    id = Column(Integer, primary_key=True, index=True)
    threat_species = Column(String)
    threat_lat = Column(Float)
    threat_lon = Column(Float)
    safe_zone_name = Column(String)
    safe_zone_lat = Column(Float)
    safe_zone_lon = Column(Float)
    recommended_heading = Column(String)
    distance_meters = Column(Integer)
    risk_level = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

class AdminLogin(Base):
    __tablename__ = "admin_logins"
    id = Column(Integer, primary_key=True, index=True)
    officer_id = Column(String, index=True)
    login_time = Column(DateTime, default=datetime.now)
    ip_address = Column(String)

class SystemSetting(Base):
    __tablename__ = "system_settings"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)

class WildlifeEvidenceSubmission(Base):
    __tablename__ = "wildlife_evidence_submissions"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(String, unique=True, index=True) # UUID string
    timestamp = Column(DateTime, default=datetime.now)
    file_name = Column(String)
    file_path = Column(String)
    file_size_bytes = Column(Integer)
    mime_type = Column(String)
    sha256_hash = Column(String, index=True)
    phash = Column(String, index=True)
    
    # User-provided metadata (privacy-safe)
    suspected_animal = Column(String, default="unknown") # tiger, elephant, wild boar, other, unknown
    source_type = Column(String, default="unknown") # mobile camera, CCTV/IP camera, trail camera, downloaded image, unknown
    reported_location_name = Column(String, nullable=True)
    reported_lat = Column(Float, nullable=True)
    reported_lon = Column(Float, nullable=True)
    reported_capture_time = Column(String, nullable=True)
    description = Column(String, nullable=True)
    contact_info = Column(String, nullable=True) # Optional contact, protected
    
    # Technical & EXIF metadata (stored as JSON)
    metadata_json = Column(String, nullable=True)
    embedded_lat = Column(Float, nullable=True)
    embedded_lon = Column(Float, nullable=True)
    embedded_timestamp = Column(String, nullable=True)
    camera_make = Column(String, nullable=True)
    camera_model = Column(String, nullable=True)
    lens_model = Column(String, nullable=True)
    software_history = Column(String, nullable=True)
    
    # Forensics & Duplicate analysis (stored as JSON)
    forensics_json = Column(String, nullable=True)
    duplicate_json = Column(String, nullable=True)
    
    # Location verification analysis (stored as JSON)
    location_verification_json = Column(String, nullable=True)
    location_classification = Column(String, default="Location unavailable")
    
    # Wildlife AI Detection (stored as JSON)
    wildlife_detection_json = Column(String, nullable=True)
    
    # Evidence Decision Engine Outcome
    decision_outcome = Column(String, default="ACCEPTED FOR REVIEW") # ACCEPTED FOR REVIEW, FLAGGED FOR REVIEW, REJECTED, ANALYSIS INCOMPLETE
    decision_explanation = Column(String)
    flags_json = Column(String, nullable=True)
    
    # Reviewer & Audit Trail
    review_status = Column(String, default="PENDING") # PENDING, UNDER_REVIEW, VERIFIED, REJECTED
    reviewer_notes = Column(String, nullable=True)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

class EvidenceAuditLog(Base):
    __tablename__ = "evidence_audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(String, index=True)
    action = Column(String) # SUBMISSION_CREATED, STATUS_UPDATED, REVIEWER_NOTE_ADDED, REJECTED, VERIFIED
    performed_by = Column(String) # SYSTEM or username
    details = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)


