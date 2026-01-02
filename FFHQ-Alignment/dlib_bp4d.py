#!/usr/bin/env python3
import os
import csv
import argparse
import multiprocessing as mp

import cv2
import dlib
import numpy as np
from tqdm import tqdm
import PIL.Image
import scipy.ndimage
from ffhq_align import image_align


# ==============================
# CONFIG: TARGET AUs (12 of them)
# ==============================
# These are COLUMN INDICES in the AU_OCC CSV (excluding the first frame index column).
# You must set them according to your AU mapping.
# Example for common BP4D 12 AUs: [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
TARGET_AU_COLUMNS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
MISSING_LABEL = 9


# ==============================
# DLIB (INITIALIZED PER PROCESS)
# ==============================

detector = None
predictor = None

def init_dlib(shape_predictor_path):
    global detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)


def find_frame(video_dir, frame_idx):
    """
    BP4D frame names are not guaranteed to be zero-padded.
    Try raw, 3-digit, 4-digit forms.
    """
    candidates = [
        f"{frame_idx}.jpg",
        f"{frame_idx:03d}.jpg",
        f"{frame_idx:04d}.jpg"
    ]

    for name in candidates:
        path = os.path.join(video_dir, name)
        if os.path.isfile(path):
            return path

    # If still not found, brute-force fallback (slow but safe)
    # e.g., frame_idx == 5 → match files starting with '5.'
    fallback = f"{frame_idx}"
    for f in os.listdir(video_dir):
        if f.startswith(fallback) and f.lower().endswith(".jpg"):
            return os.path.join(video_dir, f)

    return None




# ==============================
# LABEL CHECK
# ==============================

def is_legally_labelled(label_values, target_cols, missing_value=MISSING_LABEL):
    """
    label_values: list/array of label ints (length ~100) for a frame (WITHOUT frame index).
    target_cols: indices into label_values for the 12 AUs.
    Rule: a frame is legal if *all 12 target AUs* are not 'missing_value' (9).
    """
    try:
        vals = [label_values[c] for c in target_cols]
    except IndexError:
        # If mapping is wrong / out of range, treat as illegal
        return False
    return all(v != missing_value for v in vals)


# ==============================
# PER-FRAME PROCESS
# ==============================

def process_frame(src_path, dst_path):
    """
    Per-frame pipeline:
    - Read image
    - Detect face
    - Predict 68 landmarks
    - FFHQ align
    Returns (ok: bool, src_path: str)
    """
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False, src_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) == 0:
            return False, src_path

        # Use largest face (if multiple)
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(gray, face)
        landmarks = [(pt.x, pt.y) for pt in shape.parts()]

        image_align(src_path, dst_path, landmarks)
        return True, src_path

    except Exception:
        return False, src_path


# ==============================
# JOB COLLECTION
# ==============================

def collect_jobs(occ_root, video_root, output_root):
    """
    Scan AU_OCC CSVs and generate jobs (src_path, dst_path)
    for legally labelled frames only.
    """
    jobs = []

    for fname in sorted(os.listdir(occ_root)):
        if not fname.endswith(".csv"):
            continue

        # Parse subject & task from filename, e.g. F001_T1.csv
        base = os.path.splitext(fname)[0]  # F001_T1
        parts = base.split("_")
        if len(parts) != 2:
            continue
        subject, task = parts[0], parts[1]  # F001, T1

        csv_path = os.path.join(occ_root, fname)
        video_dir = os.path.join(video_root, subject, task)

        if not os.path.isdir(video_dir):
            # No corresponding video dir
            continue

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header row 0,1,2,...,99

            for row in reader:
                if not row:
                    continue
                # first value: frame index; remaining values: labels
                frame_idx = int(row[0]) - 1
                labels = [int(x) for x in row]

                if not is_legally_labelled(labels, TARGET_AU_COLUMNS):
                    continue

                src_path = find_frame(video_dir, frame_idx)
                if src_path is None:
                    print(video_dir,frame_idx)
                    print('frame missing')
                    continue

                # output path mirrors subject/task/frame name

                frame_name = str(frame_idx) + '.jpg'
                dst_path = os.path.join(output_root, subject, task, frame_name)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                jobs.append((src_path, dst_path))

    return jobs


# ==============================
# MAIN
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Check legally labelled BP4D frames and align with dlib + FFHQ."
    )
    parser.add_argument(
        "--occ_root",
        type=str,
        default = '/CIL_PROJECTS/AU_dataset/BP4D/AUCoding/AU_OCC',
        help="Root of AU_OCC label CSVs, e.g. /CIL_PROJECTS/AU_dataset/BP4D/AUCoding/AU_OCC",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default= '/CIL_PROJECTS/AU_dataset/BP4D/videos',
        help="Root of BP4D videos, e.g. /CIL_PROJECTS/AU_dataset/BP4D/videos",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default = './dlib_bp4d',
        help="Output directory for aligned images.",
    )
    parser.add_argument(
        "--shape_predictor",
        type=str,
        default="../FG-Net-main/face_align/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-face-landmark model.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for multiprocessing.",
    )
    args = parser.parse_args()

    print("Collecting jobs from AU_OCC CSVs...")
    jobs = collect_jobs(args.occ_root, args.video_root, args.output_root)
    print(f"Total legally-labelled frames found: {len(jobs)}")

    if not jobs:
        print("No jobs found. Check TARGET_AU_COLUMNS or paths.")
        return

    # Multiprocessing pool: dlib initialized per worker
    with mp.Pool(
        processes=args.workers,
        initializer=init_dlib,
        initargs=(args.shape_predictor,),
    ) as pool:
        results = list(
            tqdm(
                pool.starmap(process_frame, jobs, chunksize=50),
                total=len(jobs),
                desc="Aligning frames",
            )
        )

    # Collect failures
    failed = [src for ok, src in results if not ok]
    print(f"Finished. Success: {len(results) - len(failed)}, Failed: {len(failed)}")

    if failed:
        failed_log = os.path.join(args.output_root, "failed_frames.txt")
        with open(failed_log, "w") as f:
            f.write("\n".join(failed))
        print(f"Failed frame list saved to: {failed_log}")


if __name__ == "__main__":
    main()
