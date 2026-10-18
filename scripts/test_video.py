import cv2
import sys

video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open video: {video_path}")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 50)
print("VIDEO INFORMATION")
print("=" * 50)
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps}")
print(f"Frames     : {frame_count}")

if fps > 0:
    print(f"Duration   : {frame_count / fps:.2f} seconds")

print("=" * 50)

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not decode the first frame.")
    cap.release()
    sys.exit(1)

print("First frame decoded successfully.")
print("Frame shape:", frame.shape)

cap.release()
print("Video test successful.")
