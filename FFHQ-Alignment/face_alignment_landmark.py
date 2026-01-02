import os
import argparse
import numpy as np
import face_alignment
from skimage import io
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import torch

VALID_EXT = {".jpg"}


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in VALID_EXT


def list_images(root):
    img_list = []
    for r, _, files in os.walk(root):
        for f in files:
            if is_image_file(f):
                img_list.append(os.path.join(r, f))
    return img_list


def init_fa():
    """Each worker initializes its own FA model (avoids GPU contention)."""
    global FA_MODEL
    FA_MODEL = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def process_one(img_path, output_root):
    rel = os.path.relpath(img_path, start=root_dir)
    save_path = os.path.join(output_root, rel)
    save_path = os.path.splitext(save_path)[0] + ".npy"
    if os.path.exists(save_path):
        return img_path, "ok"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        img = io.imread(img_path)
        preds = FA_MODEL.get_landmarks(img)

        if preds is None:
            return img_path, "no_face"

        np.save(save_path, preds[0])  # save first detected face landmarks
        return img_path, "ok"

    except Exception as e:
        return img_path, f"error:{str(e)}"


def batch_process(image_paths, output_root, num_workers):
    with Pool(num_workers, initializer=init_fa) as pool:
        worker_fn = partial(process_one, output_root=output_root)
        results = list(tqdm(pool.imap(worker_fn, image_paths), total=len(image_paths)))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disfa_root", type=str, default = 'FG-Net-main/data/DISFA/aligned_images_source')
    parser.add_argument("--bp4d_root", type=str, default = 'FG-Net-main/data/BP4D/aligned_images_ori')
    parser.add_argument("--output_dir", type=str,  default = 'FG-Net-main/data/DISFA/aligned_images_face_alignment')
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Collect all image paths ---
    #disfa_imgs = list_images(args.disfa_root)
    bp4d_imgs = list_images(args.bp4d_root)

    #print(f"DISFA images: {len(disfa_imgs)}")
    print(f"BP4D images:  {len(bp4d_imgs)}")

    #all_imgs = disfa_imgs + bp4d_imgs
    global root_dir
    root_dir = os.path.commonpath([args.disfa_root, args.bp4d_root])

    # --- Run extraction ---
    results = batch_process(bp4d_imgs, args.output_dir, args.workers)

    # --- Save failed log ---
    failed = [r for r in results if r[1] != "ok"]
    if failed:
        with open(os.path.join(args.output_dir, "failed.txt"), "w") as f:
            for img, reason in failed:
                f.write(f"{img}\t{reason}\n")
        print(f"Failed images: {len(failed)} (saved to failed.txt)")
    else:
        print("All images processed successfully!")
