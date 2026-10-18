import cv2
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python scripts/create_low_res.py <input_video> <output_video>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"ERROR: Input video does not exist: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        sys.exit(1)

    # Original video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2x downsampling
    scale = 2

    lr_width = width // scale
    lr_height = height // scale

    print("=" * 60)
    print("LOW-RESOLUTION VIDEO CREATION")
    print("=" * 60)
    print(f"Input resolution : {width} x {height}")
    print(f"Output resolution: {lr_width} x {lr_height}")
    print(f"Scale factor     : {scale}x")
    print(f"FPS              : {fps}")
    print(f"Input frames     : {frame_count}")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (lr_width, lr_height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        sys.exit(1)

    processed_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Downsample using area interpolation.
        lr_frame = cv2.resize(
            frame,
            (lr_width, lr_height),
            interpolation=cv2.INTER_AREA
        )

        writer.write(lr_frame)

        processed_frames += 1

    cap.release()
    writer.release()

    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {processed_frames}")
    print(f"Output          : {output_path}")

    if processed_frames == frame_count:
        print("Frame count verification: PASS")
    else:
        print(
            f"Frame count verification: WARNING "
            f"(expected {frame_count}, got {processed_frames})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
