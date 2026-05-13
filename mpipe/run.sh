#!/bin/bash

# MediaPipe Pose and Face Detection - Run Script
# This script runs the MediaPipe processing pipeline on a video file

# Configuration
INPUT_VIDEO="source_video.mp4"
OUTPUT_VIDEO="output_video.mp4"
OUTPUT_JSON="output.json"
FRAMES_DIR="frames"
NO_DISPLAY=false  # Set to true to disable display window

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d "mp_env" ]; then
    echo "Activating virtual environment (mp_env)..."
    source mp_env/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Upgrade pip and install dependencies
echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video '$INPUT_VIDEO' not found!"
    echo "Please update INPUT_VIDEO in run.sh or provide a valid video file."
    exit 1
fi

# Build the command
CMD="python mpipe.py --input $INPUT_VIDEO --output_video $OUTPUT_VIDEO --output_json $OUTPUT_JSON --frames_dir $FRAMES_DIR"

if [ "$NO_DISPLAY" = true ]; then
    CMD="$CMD --no_display"
fi

# Run the script
echo "Starting MediaPipe processing..."
echo "Input: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSON: $OUTPUT_JSON"
echo "Frames directory: $FRAMES_DIR"
echo ""
echo "Running: $CMD"
echo ""

$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Processing completed successfully!"
    echo "Check the output files:"
    echo "  - $OUTPUT_VIDEO"
    echo "  - $OUTPUT_JSON"
    echo "  - $FRAMES_DIR/ (directory with PNG frames)"
else
    echo ""
    echo "Processing failed with errors."
    exit 1
fi

