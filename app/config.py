import os
from pathlib import Path

# Base directory of the project (one level up from app/)
BASE_DIR = Path(__file__).resolve().parent.parent

# CORS
ALLOWED_ORIGINS = [
    "https://burgerhotdog.github.io",
    "http://localhost:5173",
]

# OCR
TESSERACT_CONFIG = r"--oem 3 --psm 7"
FUZZY_MATCH_CUTOFF = 0.8
EXPECTED_IMAGE_SIZE = (1920, 1080)
TEMPLATE_MATCH_THRESHOLD = 0.8

# Template image path
NAME_LV_PATH = BASE_DIR / "nameLV.webp"
