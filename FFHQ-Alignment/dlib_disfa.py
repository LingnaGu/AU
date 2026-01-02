import os
import cv2
import dlib
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import argparse
import PIL.Image
import scipy.ndimage

# ---------------------------
# FFHQ ALIGNMENT FUNCTION
# ---------------------------

def image_align(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17, :2]  # left-right
        lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
        lm_eyebrow_right = lm[22 : 27, :2]  # left-right
        lm_nose          = lm[27 : 31, :2]  # top-down
        lm_nostrils      = lm[31 : 36, :2]  # top-down
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file, 'PNG')


# ---------------------------
# INIT DLIB IN EACH WORKER
# ---------------------------

def init_dlib(predictor_path):
    global detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)


# ---------------------------
# PROCESS SINGLE IMAGE
# ---------------------------

def process_one_image(src_path, dst_path):
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False, src_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) == 0:
            return False, src_path

        shape = predictor(gray, faces[0])
        landmarks = [(pt.x, pt.y) for pt in shape.parts()]

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        image_align(src_path, dst_path, landmarks)

        return True, src_path

    except Exception as e:
        return False, src_path


# ---------------------------
# MULTIPROCESS PIPELINE
# ---------------------------

def run_multiprocess(input_root, output_root, predictor_path, num_workers):
    all_images = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(root, f)
                rel = os.path.relpath(src, input_root)
                dst = os.path.join(output_root, rel)
                all_images.append((src, dst))

    print("Total images:", len(all_images))

    with mp.Pool(
        processes=num_workers,
        initializer=init_dlib,
        initargs=(predictor_path,)
    ) as pool:
        results = list(
            tqdm(pool.starmap(process_one_image, all_images, chunksize=50),
                 total=len(all_images))
        )

    failed = [p for ok, p in results if not ok]
    print(f"Finished. Success = {len(results) - len(failed)}, Failed = {len(failed)}")

    if failed:
        with open("failed_images_dlib_disfa.txt", "w") as f:
            f.write("\n".join(failed))
# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default = '/CIL_PROJECTS/AU_dataset/DISFA/raw_frames' , help="Path to DISFA raw frames")
    parser.add_argument("--output", default = './DISFA_dlib', help="Output aligned directory")
    parser.add_argument("--shape_predictor", default="../FG-Net-main/face_align/shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    run_multiprocess(
        input_root=args.input,
        output_root=args.output,
        predictor_path=args.shape_predictor,
        num_workers=args.workers
    )
