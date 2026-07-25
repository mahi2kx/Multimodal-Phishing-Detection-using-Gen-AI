"""
config.py
---------
Central configuration for the Multimodal Phishing Detection pipeline.
Edit these paths/settings once; all three pipeline scripts import from here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CSV = os.path.join(BASE_DIR, "dataset", "urls.csv")   # input: url,label
HTML_OUTPUT_DIR = os.path.join(BASE_DIR, "Alloutputs")        # part1 output
IMAGES_OUTPUT_DIR = os.path.join(BASE_DIR, "images")          # part2 output
OCR_OUTPUT_CSV = os.path.join(BASE_DIR, "ocr_output.csv")     # part2 output (image -> ocr text)

FEATURES_OUTPUT_XLSX = os.path.join(BASE_DIR, "features_output.xlsx")  # part3 output
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "phishing_clf.joblib")      # part3 output
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "embeddings_cache.npz")      # part3 cache

# ---------------------------------------------------------------------------
# Part 1: HTML Downloader
# ---------------------------------------------------------------------------
HTML_REQUEST_TIMEOUT = 15          # seconds
HTML_MAX_WORKERS = 20              # concurrent threads
HTML_RETRY_COUNT = 2

# ---------------------------------------------------------------------------
# Part 2: Image Scraper + OCR
# ---------------------------------------------------------------------------
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200
PAGE_LOAD_TIMEOUT_MS = 20000        # Playwright timeout in ms
OCR_LANGUAGES = ["en"]              # EasyOCR languages
IMAGE_SCRAPER_CONCURRENCY = 5       # parallel browser contexts

# ---------------------------------------------------------------------------
# Part 3: Feature Extraction + Training
# ---------------------------------------------------------------------------
ROBERTA_MODEL_NAME = "roberta-base"
MAX_TOKEN_LENGTH = 256
EMBEDDING_BATCH_SIZE = 16
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# Device selection (auto-detects GPU if available; falls back to CPU)
USE_GPU_IF_AVAILABLE = True
