"""
Configuration settings for Structure from Motion pipeline.
"""
from pathlib import Path

# Directory configuration
DATA_DIR = Path("data")
# OUT_DIR = Path("runs/run_3595")
OUT_DIR = Path("runs/run_3592")

# Video processing configuration
TARGET_FPS = 7.0  # Target frames per second for frame extraction (8)

# Feature detection configuration
METHOD = "sift"  # {"sift", "orb"} - Feature detection method
RATIO = 0.8      # Lowe ratio for feature matching (more permissive)
CROSSCHECK = True  # Enable symmetric Lowe-ratio check for better matches

# Visualization configuration
MAX_VIZ = 200    # Maximum number of matches to visualize

# SfM pipeline configuration
MIN_MATCHES = 20  # Minimum number of matches required for pose estimation (further reduced)
CONFIDENCE = 0.90  # Confidence level for fundamental matrix estimation (further reduced)
