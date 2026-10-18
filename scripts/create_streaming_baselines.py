import subprocess
import sys
from pathlib import Path


def run_ffmpeg(command):
    print("\nRunning:")
    print(" ".join(command))
    print()

    result = subprocess.run(command)

    if result.returncode != 0:
        print("ERROR: FFmpeg command failed.")
        sys.exit(result.returncode)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/create_streaming_baselines.py <input_video>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"ERROR: Input video does not exist: {input_path}")
        sys.exit(1)

    hr_output = Path("videos/streaming/hr_640x360.mp4")
    lr_output = Path("videos/streaming/lr_320x180.mp4")

    hr_output.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # HR STREAMING VERSION
    # ---------------------------------------------------------

    print("=" * 60)
    print("CREATING HR STREAMING VIDEO")
    print("=" * 60)

    hr_command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),

        "-vf", "scale=640:360",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",

        "-an",

        str(hr_output)
    ]

    run_ffmpeg(hr_command)

    # ---------------------------------------------------------
    # LR STREAMING VERSION
    # ---------------------------------------------------------

    print("=" * 60)
    print("CREATING LR STREAMING VIDEO")
    print("=" * 60)

    lr_command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),

        "-vf", "scale=320:180",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",

        "-an",

        str(lr_output)
    ]

    run_ffmpeg(lr_command)

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("CONTROLLED ENCODING COMPLETE")
    print("=" * 60)

    print(f"HR output: {hr_output}")
    print(f"LR output: {lr_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
