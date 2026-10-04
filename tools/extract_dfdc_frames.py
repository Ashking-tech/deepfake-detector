"""DFDC videos -> notebook-ready frames (train/val/test + real/fake).

Plain story: the DFDC Preview set is folders of mp4s plus a metadata file
per folder saying which video is FAKE or REAL. This script reads N evenly
spaced frames from every video and sorts them into the exact folder layout
training/train_heads.ipynb expects.

Needs OpenCV only (`pip install opencv-python`). In Colab cv2 is already
there. Run it with:
    python tools/extract_dfdc_frames.py --input /path/to/dfdc_preview --output /path/to/frames
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Make `app` importable when run as a script (same bootstrap as the rest).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# How many frames to pull from each video (evenly spaced, middle-biased).
FRAMES_PER_VIDEO = 10

# Video list split (by video, never by frame — same face must not leak
# across splits or the AUC numbers lie).
TRAIN_SHARE = 0.80
VAL_SHARE = 0.10

# Longest side of saved frames (keeps Drive + Colab RAM happy; the
# pipeline center-crops to 224 anyway, so nothing learned is lost).
MAX_SIDE = 512

# JPEG quality for saved frames (high: we must not add OUR OWN
# compression fingerprints on top of the dataset's).
JPEG_QUALITY = 95

# Fixed shuffle seed so re-runs produce the identical split.
SPLIT_SEED = 7


# Reads every metadata.json under root: filename -> "FAKE"/"REAL".
def load_labels(root: Path) -> dict:
    """Walk root for metadata files, return {video_path: label}."""
    # Step 1: find every metadata file (one per DFDC folder).
    labels = {}

    for meta_path in sorted(root.rglob("*.json")):
        # Step 2: skip anything that is not a DFDC metadata file.
        try:
            entries = json.loads(meta_path.read_text())
        except Exception:
            continue

        if not isinstance(entries, dict):
            continue

        # Step 3: keep entries that look like {"file.mp4": {"label": ...}}.
        for name, info in entries.items():
            if not isinstance(info, dict):
                continue

            label = str(info.get("label", "")).upper()

            if label not in ("FAKE", "REAL"):
                continue

            video_path = meta_path.parent / name

            if video_path.exists():
                labels[str(video_path)] = label

    return labels


# Splits video paths into train/val/test by video (seeded shuffle).
def split_videos(paths: list) -> dict:
    """Return {split: [paths]} with an 80/10/10 video-level split."""
    # Step 1: seeded shuffle (same split every run).
    ordered = sorted(paths)

    rng = random.Random(SPLIT_SEED)

    rng.shuffle(ordered)

    # Step 2: cut into three chunks.
    total = len(ordered)

    train_end = int(total * TRAIN_SHARE)

    val_end = train_end + int(total * VAL_SHARE)

    splits = {}

    splits["train"] = ordered[:train_end]

    splits["val"] = ordered[train_end:val_end]

    splits["test"] = ordered[val_end:]

    return splits


# Pulls N evenly spaced frames from one video, saves them to out_dir.
# Returns the number of frames actually saved.
def extract_video(video_path: str, out_dir: Path, label: str, stem: str) -> int:
    """Sample frames with cv2, downscale, save as JPEGs. Returns count."""
    import cv2

    # Step 1: open the video.
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  skip (unreadable): {video_path}")

        return 0

    # Step 2: count frames, pick N evenly spaced indexes.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()

        return 0

    picks = [int(total * (i + 1) / (FRAMES_PER_VIDEO + 1)) for i in range(FRAMES_PER_VIDEO)]

    # Step 3: grab, downscale, save each picked frame.
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for rank, index in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame = cap.read()

        if not ok:
            continue

        height, width = frame.shape[:2]

        longest = max(height, width)

        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest

            new_width = int(width * scale)

            new_height = int(height * scale)

            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        out_name = f"{stem}_{rank:02d}.jpg"

        cv2.imwrite(str(out_dir / out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        saved += 1

    cap.release()

    return saved


def main() -> None:
    """Parse args, split videos, extract frames into notebook layout."""
    global FRAMES_PER_VIDEO

    # Step 1: read the command line.
    parser = argparse.ArgumentParser(description="DFDC videos -> frame folders")

    parser.add_argument("--input", required=True, help="DFDC preview root (mp4s + metadata)")

    parser.add_argument("--output", required=True, help="frames root (notebook DATA_ROOT)")

    parser.add_argument("--frames", type=int, default=FRAMES_PER_VIDEO, help="frames per video")

    args = parser.parse_args()

    FRAMES_PER_VIDEO = args.frames

    # Step 2: load labels and split by video.
    labels = load_labels(Path(args.input))

    n_fake = sum(1 for label in labels.values() if label == "FAKE")

    print(f"videos: {len(labels)} ({n_fake} fake)")

    if not labels:
        print("no labeled videos found — check --input points at the DFDC root")

        return

    splits = split_videos(list(labels.keys()))

    # Step 3: extract every video into split/label/ folders.
    grand_total = 0

    for split in ["train", "val", "test"]:
        for video_path in splits[split]:
            label = labels[video_path]

            class_name = "fake" if label == "FAKE" else "real"

            out_dir = Path(args.output) / split / class_name

            stem = Path(video_path).stem

            saved = extract_video(video_path, out_dir, label, stem)

            grand_total += saved

        n_split = sum(1 for _ in (Path(args.output) / split).rglob("*.jpg"))

        print(f"{split}: {n_split} frames")

    print(f"done: {grand_total} frames under {args.output}")


if __name__ == "__main__":
    main()
