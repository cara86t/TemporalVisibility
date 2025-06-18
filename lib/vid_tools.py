"""Provides tools for processing videos."""

__version__ = "0.1"
__author__ = "Cara Tursun"
__copyright__ = """Copyright (c) 2023 Cara Tursun"""

import numpy as np


def read_video_file(filename, frame_fn=lambda x: x, framen=0):
    """Read the video, returns as a
    numpy array of size (framen, height, width)"""
    from decord import VideoReader, VideoLoader
    from decord import cpu, gpu

    # avoiding the use of GPU here because it may interact
    # with the FPS stability during the experiment
    vr = VideoReader(filename, ctx=cpu(0))
    print(f"# of frames in {filename}: {len(vr)}")
    if framen > 0:
        for i in range(len(vr) // framen - 1):
            yield vr.get_batch(range(i * framen, (i + 1) * framen)).asnumpy()
    else:
        return vr.get_batch(range(len(vr))).asnumpy()


def apply_frame_func(video, frame_fn=lambda x: x):
    result = None
    for frame in video:
        if result is None:
            result = frame_fn(frame)
            result = np.reshape(result, (1,) + result.shape)
        else:
            newframe = frame_fn(frame)
            newframe = np.reshape(newframe, (1,) + newframe.shape)
            result = np.concatenate((result, newframe), axis=0)
    return result


def save_video_file(video, filename, FPS=120):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if filename.lower().endswith(".mp4"):
        ext = ""
    else:
        ext = ".mp4"
    out = cv2.VideoWriter(
        filename + ext, fourcc, FPS, (video.shape[2], video.shape[1])
    )
    for frame in video:
        out.write(frame[:, :, ::-1])
    out.release()


def npy2mp4(directory):
    import os

    for root, _, files in os.walk(directory):
        if root != directory:
            continue
        for file in files:
            clip = np.load(os.path.join(root, file))
            filename, _ = os.path.splitext(file)
            save_video_file(
                clip, os.path.join(root, filename + ".mp4"), FPS=120
            )


if __name__ == "__main__":
    pass
    # npy2mp4("vid_segments/10")
