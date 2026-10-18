import cv2
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/video_info.py <video>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    file_size = video_path.stat().st_size

    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0

    if duration > 0:
        bitrate_bps = (file_size * 8) / duration
    else:
        bitrate_bps = 0

    print("=" * 60)
    print("VIDEO INFORMATION")
    print("=" * 60)

    print(f"File       : {video_path}")
    print(f"File size  : {file_size / 1024:.2f} KB")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps}")
    print(f"Frames     : {frame_count}")
    print(f"Duration   : {duration:.2f} seconds")
    print(f"Bitrate    : {bitrate_bps / 1000:.2f} kbps")

    print("=" * 60)


if __name__ == "__main__":
    main()
