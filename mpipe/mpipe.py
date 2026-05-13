import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
import argparse

#arguments
parser = argparse.ArgumentParser(description="MediaPipe implementation")
parser.add_argument("--input", required=True, help="Path to input video")
parser.add_argument("--output_video", default="output_video.mp4", help="Output video path")
parser.add_argument("--output_json", default="output.json", help="Output keypoints JSON")
parser.add_argument("--frames_dir", default="frames", help="Directory to save PNG frames")
parser.add_argument("--no_display", action="store_true", help="Disable cv2.imshow")
args = parser.parse_args()

#config
SOURCE_VIDEO = args.input
OUTPUT_VIDEO = args.output_video
OUTPUT_JSON = args.output_json
FRAMES_DIR = Path(args.frames_dir)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

DRAW_FACE = True
DRAW_POSE = True

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(SOURCE_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {SOURCE_VIDEO}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

#pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity = 2,
    enable_segmentation=False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

#face
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
)

all_frames = []
frame_id = 0

def connect_face_pose(canvas, pose_lms, face_lms, w, h):
    # Pose nose
    pose_nose = pose_lms.landmark[mp_pose.PoseLandmark.NOSE]

    # Face nose tip (stable)
    face_nose = face_lms.landmark[1]

    p1 = (int(pose_nose.x * w), int(pose_nose.y * h))
    p2 = (int(face_nose.x * w), int(face_nose.y * h))

    cv2.line(canvas, p1, p2, (0, 255, 0), 2)
    print("Connected face and pose")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    black = np.zeros((height, width, 3), dtype=np.uint8)

    pose_results = pose.process(rgb)
    face_results = face_mesh.process(rgb)

    frame_data = {
        "frame_id": frame_id,
        "pose_landmarks": [],
        "face_landmarks": []
    }

    #pose
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            x = lm.x * width
            y = lm.y * height
            frame_data["pose_landmarks"].append([x, y, lm.visibility])
        
        if DRAW_POSE:
            mp_draw.draw_landmarks(
                black, 
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    
    #face
    face_landmarks = None
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        for lm in face_landmarks.landmark:
            x = lm.x * width
            y = lm.y * height
            frame_data["face_landmarks"].append([x, y, lm.visibility])
        
        if DRAW_FACE:
            mp_draw.draw_landmarks(
                black, 
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
    
    if pose_results.pose_landmarks is not None and face_landmarks is not None:
        connect_face_pose(
            black,
            pose_results.pose_landmarks,
            face_landmarks,
            width,
            height
        )
    else:
        print("No pose or face landmarks")
    
    all_frames.append(frame_data)
    out.write(black)

    frame_path = FRAMES_DIR / f"{frame_id:06d}.png"
    cv2.imwrite(str(frame_path), black)

    if not args.no_display:
        cv2.imshow("frame", black)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

with open(OUTPUT_JSON, 'w') as f:
    json.dump(all_frames, f, indent=4)

print(f'Completed, processed {frame_id} frames')