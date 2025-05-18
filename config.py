"""
Configuration file for database and other settings.
This file should NOT be committed to GitHub as it contains sensitive information.
"""

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password_here",  # Replace with your actual password when using locally
    "database": "vehicle_data_callback"
}

# Video settings
VIDEO_PATH = "path/to/your/video.mp4"  # Replace with your actual video path

# Model settings
MODEL_PATH = "yolov8n.pt"  # Default model

# Line and distance settings
DISTANCE_METERS = 10  # Distance between entry and exit lines in meters