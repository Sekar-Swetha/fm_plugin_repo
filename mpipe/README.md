# MediaPipe Pose and Face Detection

A Python application that processes video files to detect and visualize pose and face landmarks using Google's MediaPipe library.

## Features

- **Pose Detection**: Detects 33 body pose landmarks with full body pose estimation
- **Face Mesh Detection**: Detects 468 facial landmarks with refined landmark detection
- **Landmark Visualization**: Draws pose and face landmarks on video frames
- **Face-Pose Connection**: Connects the face and pose landmarks via a green line
- **Output Formats**: 
  - Output video with visualized landmarks
  - JSON file containing all landmark coordinates
  - Individual PNG frames for each processed frame

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- MediaPipe

## Installation

1. Clone or navigate to this repository:
   ```bash
   cd mpipe
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. Install required packages:
   ```bash
   pip install opencv-python numpy mediapipe
   ```

   Or if you have a requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python mpipe.py --input <path_to_video>
```

### Command Line Arguments

- `--input` (required): Path to the input video file
- `--output_video` (optional): Output video path (default: `output_video.mp4`)
- `--output_json` (optional): Output JSON file path (default: `output.json`)
- `--frames_dir` (optional): Directory to save PNG frames (default: `frames`)
- `--no_display` (optional): Disable the display window (useful for headless servers)

### Examples

1. Process a video with default settings:
   ```bash
   python mpipe.py --input source_video.mp4
   ```

2. Process a video with custom output paths:
   ```bash
   python mpipe.py --input source_video.mp4 --output_video my_output.mp4 --output_json my_keypoints.json
   ```

3. Process a video without displaying the preview window:
   ```bash
   python mpipe.py --input source_video.mp4 --no_display
   ```

4. Process a video and save frames to a custom directory:
   ```bash
   python mpipe.py --input source_video.mp4 --frames_dir my_frames
   ```

### Using the Run Script

Alternatively, you can use the provided run script:

```bash
chmod +x run.sh
./run.sh
```

Edit `run.sh` to customize the input video path and other options.

## Output

The script generates three types of output:

1. **Output Video** (`output_video.mp4`): Video with pose and face landmarks drawn on a black background
   - Red points and green connections for pose landmarks
   - Green mesh for face landmarks
   - Green line connecting pose nose to face nose tip

2. **JSON File** (`output.json`): Contains landmark data for each frame:
   ```json
   [
     {
       "frame_id": 0,
       "pose_landmarks": [[x, y, visibility], ...],
       "face_landmarks": [[x, y, z], ...]
     },
     ...
   ]
   ```

3. **PNG Frames** (`frames/` directory): Individual PNG images for each processed frame, named as `000000.png`, `000001.png`, etc.

## Configuration

You can modify the following settings in `mpipe.py`:

- `DRAW_FACE`: Set to `False` to disable face landmark drawing (default: `True`)
- `DRAW_POSE`: Set to `False` to disable pose landmark drawing (default: `True`)
- `model_complexity`: Pose model complexity (0, 1, or 2) - higher is more accurate but slower (default: `2`)
- `min_detection_confidence`: Minimum confidence for pose detection (default: `0.5`)
- `min_tracking_confidence`: Minimum confidence for pose tracking (default: `0.5`)

## Notes

- The script processes videos frame by frame, so processing time depends on video length and resolution
- Press 'q' during playback to quit early
- The script maintains the original video's resolution and FPS
- Face detection is limited to 1 face per frame
- If no pose or face landmarks are detected in a frame, the frame will still be processed but without landmark drawings

## Troubleshooting

- **Video won't open**: Ensure the video path is correct and the video format is supported by OpenCV
- **Low performance**: Try reducing `model_complexity` from 2 to 1 or 0
- **Display issues**: Use `--no_display` flag if running on a headless server
- **Import errors**: Ensure all dependencies are installed in your active virtual environment

## License

This project is for dissertation research purposes.

