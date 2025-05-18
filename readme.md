# Vehicle Detection and Tracking System

A computer vision application for detecting, tracking, and monitoring vehicles in video streams. The system tracks vehicle movement, calculates speeds, identifies stationary vehicles, and detects improper parking.
<img src="sample pictures/1.png" alt="Vehicle Detection" width="400"/>

## Features

- Vehicle detection and classification using YOLOv8  
- Speed calculation for vehicles crossing designated areas  
- Stationary vehicle detection  
- Improper parking detection in restricted zones  
- Database integration for storing vehicle data  
- Logging system for monitoring application performance  
- Interactive line and restricted area drawing  

## Requirements

- Python 3.8+  
- OpenCV  
- Ultralytics YOLOv8  
- MySQL Connector  
- NumPy  

## Installation

Clone the repository:  
git clone https://github.com/your-username/vehicle-detection.git  
cd vehicle-detection  

Install required packages:  
pip install -r requirements.txt  

Set up the MySQL database:

Create a database named vehicle_data_callback  
Create a table with the following structure:

CREATE TABLE vehicle_data (  
 id INT AUTO_INCREMENT PRIMARY KEY,  
 image_path VARCHAR(255),  
 saved_time DATETIME,  
 speed FLOAT  
);

Configure the database connection in config.py (see Configuration section)

## Configuration

Create a config.py file with your database credentials:  
(Do NOT upload this file to GitHub; add it to .gitignore)

DB_CONFIG = {  
 "host": "localhost",  
 "user": "your_username",  
 "password": "your_password",  
 "database": "vehicle_data_callback"  
}

## Usage

Run the main script with your video file:  
python vehicle_tracker.py

Controls:

Press 'q' to quit  
Press 's' to set entry and exit lines (click twice on the video)  
Press 'r' to define a restricted parking area (click 4 points)

## Project Structure

vehicle-detection/  
├── vehicle_tracker.py     # Main application file  
├── config.py              # Configuration file (not tracked by git)  
├── requirements.txt       # Required Python packages  
├── README.md              # Project documentation  
├── logs_with_callback/    # Log files directory (not tracked by git)  
└── captured_frames/       # Saved vehicle images (not tracked by git)  

