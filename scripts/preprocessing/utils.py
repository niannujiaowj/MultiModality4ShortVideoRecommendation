import math
import numpy as np
import cv2


def __crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Utilities to open video files using CV2
def load_video(video_path, max_frames=32, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = __crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    frames = np.array(frames)
    if len(frames) < max_frames:
        n_repeat = int(math.ceil(max_frames / float(len(frames))))
        frames = frames.repeat(n_repeat, axis=0)
    frames = frames[:max_frames]
    return frames / 255.0


