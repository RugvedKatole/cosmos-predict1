
from io import BytesIO

import imageio
import numpy as np


def load_from_fileobj(filepath: str, format: str = "mp4", mode: str = "rgb", **kwargs):
    """
    Load video from a file-like object using imageio with specified format and color mode.

    Parameters:
        file (IO[bytes]): A file-like object containing video data.
        format (str): Format of the video file (default 'mp4').
        mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

    Returns:
        tuple: A tuple containing an array of video frames and metadata about the video.
    """
    with open(filepath, "rb") as f:
        value = f.read()
    with BytesIO(value) as f:
        f.seek(0)
        video_reader = imageio.get_reader(f, format, **kwargs)

        video_frames = []
        for frame in video_reader:
            if mode == "gray":
                import cv2  # Convert frame to grayscale if mode is gray

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=2)  # Keep frame dimensions consistent
            video_frames.append(frame)

    return np.array(video_frames), video_reader.get_meta_data()