"""
scripts/extract_frames.py
==========================
Extract frames from student enrollment videos into per-student folders.

VIDEO NAMING CONVENTION:
  The last two underscore-separated parts of the filename (before .mp4)
  are the student's first and last name.

  Examples (all produce "John Doe"):
    2024_batch_CSE_John_Doe.mp4
    enrollment_sec_B_John_Doe.mp4
    rec_001_John_Doe.mp4
    John_Doe.mp4

  The script ignores everything before the last two parts.

INPUT — drop all videos anywhere inside a single folder:
  videos/
    ├── 2024_batch_John_Doe.mp4
    ├── rec_Jane_Smith.mp4
    └── ...

OUTPUT — per-student frame folders created inside data/enrollment/:
  data/enrollment/
    ├── John_Doe/
    │   ├── frame_0000.jpg
    │   ├── frame_0005.jpg
    │   └── ...
    ├── Jane_Smith/
    │   └── ...
    └── ...

USAGE:
  # Point at the folder containing all your .mp4 files
  python scripts/extract_frames.py --videos_dir path/to/videos/

  # Single video
  python scripts/extract_frames.py --video path/to/John_Doe.mp4

  # Optional overrides
  python scripts/extract_frames.py --videos_dir path/to/videos/ --max-frames 120 --skip-frames 5

WHY SKIP EVERY 5TH FRAME?
  At 30fps, consecutive frames differ by ~33ms and are nearly identical.
  Taking every 5th frame gives ~6fps effective, ensuring pose variation
  without flooding the embedding database with duplicates.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from config import PATHS
from src.utils import setup_logging, extract_frames_from_video


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_name_from_video(video_path: Path) -> str:
    """
    Extract student name from video filename.

    Rule: split on ' - ' (space-dash-space) and take everything AFTER it.
    The name is then stripped and spaces replaced with underscores for the
    folder name.

    Examples:
      "VID_20251213_215307 - SHIVAM SINGH"      → "SHIVAM_SINGH"
      "video_20251213_000519_edit - Satyam Kumar" → "Satyam_Kumar"
      "video - STUTI SHAH"                       → "STUTI_SHAH"
      "video_20251214_235320 - Mohul"            → "Mohul"

    Fallback (no ' - ' found): uses the full stem as-is.
    This should not happen if all your files follow the convention.
    """
    stem = video_path.stem   # filename without .mp4

    if " - " in stem:
        # Take everything after the last ' - '
        name = stem.rsplit(" - ", 1)[-1].strip()
    else:
        # Fallback — warn and use full stem
        import logging
        logging.getLogger(__name__).warning(
            f"No ' - ' found in '{stem}' — using full filename as name."
        )
        name = stem.strip()

    # Replace spaces with underscores for folder name
    # e.g. "SHIVAM SINGH" → "SHIVAM_SINGH"
    return name.replace(" ", "_")


def extract_for_video(video_path: Path,
                      enrollment_dir: Path,
                      max_frames: int = 150,
                      skip_frames: int = 5) -> tuple:
    """
    Extract frames from one video into enrollment_dir/<StudentName>/.

    Args:
        video_path:     Path to the .mp4 (or other) video file
        enrollment_dir: Root enrollment directory (data/enrollment/)
        max_frames:     Maximum frames to extract
        skip_frames:    Extract one frame every N frames

    Returns:
        (student_name, num_frames_saved)
    """
    student_name = parse_name_from_video(video_path)
    out_dir      = enrollment_dir / student_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved = extract_frames_from_video(
            str(video_path),
            str(out_dir),
            max_frames=max_frames,
            skip_frames=skip_frames,
        )
        return student_name, len(saved)
    except Exception as e:
        logging.getLogger(__name__).error(
            f"  ERROR processing {video_path.name}: {e}"
        )
        return student_name, 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract enrollment frames from student videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Name parsing rule:
  Last two underscore-parts of filename = first + last name
  e.g.  2024_CSE_B_John_Doe.mp4  →  student folder: John_Doe
        """,
    )
    parser.add_argument(
        "--videos_dir", type=str, default=None,
        help="Folder containing all student .mp4 files (searches recursively)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Single video file to process"
    )
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Max frames to extract per video (default: 150)"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=5,
        help="Extract one frame every N frames (default: 5, good for 30fps)"
    )
    parser.add_argument(
        "--enrollment_dir", type=str, default=None,
        help="Output enrollment directory (default: from config.py)"
    )
    args = parser.parse_args()

    setup_logging("INFO", PATHS["log_file"])
    logger = logging.getLogger(__name__)

    enrollment_dir = Path(args.enrollment_dir or PATHS["enrollment_dir"])
    enrollment_dir.mkdir(parents=True, exist_ok=True)

    # ── Single video mode ─────────────────────────────────────────────
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)

        name, count = extract_for_video(
            video_path, enrollment_dir,
            args.max_frames, args.skip_frames
        )
        print(f"\n✓ [{name}]  {count} frames → {enrollment_dir / name}/")
        print(f"\nNow run: python scripts/run_enrollment.py")
        return

    # ── Batch mode ────────────────────────────────────────────────────
    if not args.videos_dir:
        parser.print_help()
        print("\nError: provide --videos_dir or --video")
        sys.exit(1)

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        logger.error(f"Directory not found: {videos_dir}")
        sys.exit(1)

    # Find all videos (including subdirectories)
    all_videos = sorted([
        f for f in videos_dir.rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not all_videos:
        logger.error(
            f"No video files found in {videos_dir}\n"
            f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
        )
        sys.exit(1)

    logger.info(f"Found {len(all_videos)} video(s) in {videos_dir}")

    # Preview parsed names before extracting
    print(f"\n{'─'*55}")
    print(f"{'VIDEO FILE':<35}  {'→  STUDENT NAME'}")
    print(f"{'─'*55}")
    for v in all_videos:
        name = parse_name_from_video(v)
        print(f"  {v.name:<35}  →  {name.replace('_', ' ')}")
    print(f"{'─'*55}")

    confirm = input(f"\nExtract frames for {len(all_videos)} students? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # Extract
    print()
    total_frames = 0
    results = []

    for video_path in all_videos:
        name, count = extract_for_video(
            video_path, enrollment_dir,
            args.max_frames, args.skip_frames
        )
        results.append((name, count))
        status = f"{count} frames" if count > 0 else "FAILED"
        print(f"  [{name.replace('_',' '):<25}]  {status}")
        total_frames += count

    # Summary
    print(f"\n{'═'*55}")
    print(f"Done. {len(results)} students  |  {total_frames} total frames")
    print(f"Output: {enrollment_dir}")
    failed = [n for n, c in results if c == 0]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for n in failed:
            print(f"  ✗ {n}")
    print(f"\nNext step: python scripts/run_enrollment.py")


if __name__ == "__main__":
    main()
