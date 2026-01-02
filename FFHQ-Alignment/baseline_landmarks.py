import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm
import multiprocessing as mp

IMG_ROOT = "FFHQ-Alignment/aligned_images_ori"
LMK_ROOT = "FFHQ-Alignment/aligned_landmarks_ori"
FAILED_LIST = "failed_landmarks_disfa.txt"
PREDICTOR_PATH = "/home/gulingna/FG-Net-main/face_align/shape_predictor_68_face_landmarks.dat"

os.makedirs(LMK_ROOT, exist_ok=True)

# =========================================================
# Landmark extractor
# =========================================================
def findlandmark(img_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    image = cv2.imread(img_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None

    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    return shape


# =========================================================
# Worker function for multiprocessing
# =========================================================
def process_image(img_path):
    try:
        rel_path = os.path.relpath(img_path, IMG_ROOT)
        subj = rel_path.split(os.sep)[0]
        task = rel_path.split(os.sep)[1]
        subj_dir = os.path.join(LMK_ROOT, subj)
        os.makedirs(subj_dir, exist_ok=True)
        out_dir = os.path.join(LMK_ROOT, subj, task)
        os.makedirs(out_dir, exist_ok=True)
        
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, base + ".npy")

        # Skip existing
        if os.path.exists(out_path):
            return None

        lmk = findlandmark(img_path)
        if lmk is None:
            return img_path  # failed case

        np.save(out_path, lmk)
        return None  # success
    except Exception:
        return img_path  # any unexpected error


# =========================================================
# Collect all aligned frames
# =========================================================
def collect_all_images(root):
    all_imgs = []
    for subj in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for task in sorted(os.listdir(subj_dir)):
            task_dir = os.path.join(subj_dir,task)
            if not os.path.isdir(task_dir):
                continue
            for fname in os.listdir(task_dir):
                if fname.lower().endswith((".jpg", ".png")):
                    all_imgs.append(os.path.join(task_dir, fname))
    return all_imgs


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    all_imgs = collect_all_images(IMG_ROOT)
    print(f"Found {len(all_imgs)} aligned frames.")
    num_workers = mp.cpu_count() - 4
    print(f"Using {num_workers} parallel processes.")

    failed = []

    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_image, all_imgs), total=len(all_imgs)):
            if result is not None:
                failed.append(result)

    if failed:
        with open(FAILED_LIST, "w") as f:
            for path in sorted(failed):
                f.write(path + "\n")
        print(f"{len(failed)} frames failed. Logged to {FAILED_LIST}")
    else:
        print("All frames processed successfully — no failures.")

    print("Landmark extraction completed.")

