"""
Main entry point for the Vehicle Detection and Tracking System.
Run this script with: python main.py
"""

import os
import argparse
from vehicle_tracker import VehicleTracker
from config import VIDEO_PATH, MODEL_PATH

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Vehicle Detection and Tracking System')
    parser.add_argument('-v', '--video', type=str, default=VIDEO_PATH,
                        help='Path to video file')
    parser.add_argument('-m', '--model', type=str, default=MODEL_PATH,
                        help='Path to YOLOv8 model')
    parser.add_argument('-n', '--name', type=str, default="Video Stream",
                        help='Window name for the video')
    return parser.parse_args()

def main():
    """Main function to run the vehicle tracker."""
    args = parse_arguments()
    
    # Ensure the video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return
    
    # Create and run the tracker
    print(f"Starting vehicle tracking for '{args.video}'...")
    tracker = VehicleTracker(
        video_path=args.video,
        model_path=args.model,
        window_name=args.name
    )
    
    try:
        tracker.run()
        print("Processing complete.")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()