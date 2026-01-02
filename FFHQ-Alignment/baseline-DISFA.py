import os
import cv2
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
import PIL.Image
import scipy.ndimage
import face_alignment


def image_align(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=False):
    lm = np.array(face_landmarks)
    lm_eye_left  = lm[36:42, :2]
    lm_eye_right = lm[42:48, :2]
    lm_mouth_outer = lm[48:60, :2]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c-x-y, c-x+y, c+x+y, c+x-y])
    qsize = np.hypot(*x) * 2

    if not os.path.exists(src_file):
        print("Image missing:", src_file)
        return False

    img = PIL.Image.open(src_file)

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(img.size[0] / shrink), int(img.size[1] / shrink))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    border = max(int(qsize * 0.1), 3)
    crop = (int(min(quad[:,0])) - border, int(min(quad[:,1])) - border,
            int(max(quad[:,0])) + border, int(max(quad[:,1])) + border)
    crop = (max(crop[0], 0), max(crop[1], 0),
            min(crop[2], img.size[0]), min(crop[3], img.size[1]))

    img = img.crop(crop)
    quad -= crop[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    img.save(dst_file)
    return True



def init_detector():
    global detector
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)


def process_single(job):
    raw_img_path, aligned_img_path, output_size, transform_size = job

    if os.path.exists(aligned_img_path):
        return "skip"

    try:
        landmarks = detector.get_landmarks(raw_img_path)
    except:
        return f"detect-error: {raw_img_path}"

    if landmarks is None:
        return f"no-face: {raw_img_path}"

    ok = image_align(raw_img_path, aligned_img_path, landmarks[0], output_size, transform_size)
    return "ok" if ok else "fail"


def collect_jobs(src_dir, dst_dir, output_size, transform_size):
    jobs = []
    for subject in os.listdir(src_dir):
        subpath = os.path.join(src_dir, subject)
        if not os.path.isdir(subpath):
            continue
        out_subdir = os.path.join(dst_dir, subject)
        os.makedirs(out_subdir, exist_ok=True)

        for img_name in os.listdir(subpath):
            raw_img = os.path.join(subpath, img_name)
            dst_img = os.path.join(out_subdir, img_name)
            jobs.append((raw_img, dst_img, output_size, transform_size))
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/CIL_PROJECTS/AU_dataset/DISFA/raw_frames")
    parser.add_argument("--dst", default="./aligned_images")
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--transform_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    jobs = collect_jobs(args.src, args.dst, args.output_size, args.transform_size)

    print("Total images:", len(jobs))

    with mp.Pool(args.workers, initializer=init_detector) as pool:
        results = list(tqdm(pool.imap(process_single, jobs), total=len(jobs), desc="Aligning"))


