import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def load_pose_from_video(video_path, max_frames=None, stride=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    keypoints_list = []
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=0,
                      min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx % stride) != 0:
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                      for lm in results.pose_landmarks.landmark], dtype=np.float32)
            else:
                landmarks = np.zeros((33, 4), dtype=np.float32)
            keypoints_list.append(landmarks)
            frame_idx += 1
            if max_frames and len(keypoints_list) >= max_frames:
                break
    cap.release()
    return keypoints_list
