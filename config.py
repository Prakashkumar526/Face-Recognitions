"""Central settings for the Facial Recognition Attendance System."""
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<ROLL_NO>/<ROLL_NO>_001.jpg ...
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

ENCODINGS_FILE = MODEL_DIR / "encodings.pkl"      # face_recognition (128-d) model
CNN_MODEL_FILE = MODEL_DIR / "cnn_model.keras"    # optional Keras CNN
CNN_CLASSES_FILE = MODEL_DIR / "cnn_classes.json"
ATTENDANCE_FILE = OUTPUT_DIR / "attendance.xlsx"

HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# --- capture ---------------------------------------------------------------
CAMERA_INDEX = 0
IMAGES_PER_PERSON = 30
CAPTURE_EVERY = 5          # save one image every N frames (gives pose variety)
FACE_PADDING = 0.25        # margin around the Haar box when saving crops

# --- recognition -----------------------------------------------------------
MATCH_TOLERANCE = 0.6      # max 128-d Euclidean distance for a match (report: 0.6)
PROCESS_SCALE = 0.5        # frames are downscaled for speed before detection

# --- anti-spoofing (report: movement threshold = 60) -----------------------
SPOOF_THRESHOLD = 60       # px the face centre must move inside the window
SPOOF_WINDOW_SEC = 2.0     # compare the current position with the last N seconds

# --- CNN (report: 128x128, batch 32, 10 epochs, 80/20 split) ---------------
CNN_IMG_SIZE = (128, 128)
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 10
CNN_VAL_SPLIT = 0.2
CNN_MIN_CONFIDENCE = 0.90
