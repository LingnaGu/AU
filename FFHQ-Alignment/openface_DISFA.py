import os
import cv2
import numpy as np
import argparse
import multiprocessing as mp
from queue import Empty
from openface.landmark_detection import LandmarkDetector
from openface.face_detection import FaceDetector
from ffhq_align import image_align
import torch

os.environ["OPENFACE_LOGDIR"] = os.path.expanduser("~/openface_logs")

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)



def process_one(task):
    raw_img_path, aligned_path, output_size, transform_size, padding = task

    try:
        img_raw = cv2.imread(raw_img_path)
        cropped_face, dets = face_detector.get_face(raw_img_path)

        if dets is None or len(dets) == 0:
            print(f"[NO FACE] {raw_img_path}")
            return

        landmarks = landmark_detector.detect_landmarks(img_raw, dets)
        if landmarks is None or len(landmarks) == 0:
            print(f"[NO LANDMARK] {raw_img_path}")
            return

        face_landmarks = np.array(landmarks)[0]  # 68×2

        image_align(
            raw_img_path,
            aligned_path,
            face_landmarks,
            output_size=output_size,
            transform_size=transform_size,
            enable_padding = False
        )

        print(f"[OK] {aligned_path}")

    except Exception as e:
        print(f"[ERROR] {raw_img_path}: {e}")

def worker_loop(task_queue, gpu_id):

    if gpu_id is not None:
        # Mask to a single physical GPU for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Inside this process, that GPU becomes index 0
        device = 'cuda'
        device_ids = [0]
    else:
        device = 'cpu'
        device_ids = [-1]

    print(f"[Worker] device={device}, device_ids={device_ids}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    global face_detector, landmark_detector

    # FaceDetector: only understands cpu / cuda and uses the *current* default GPU
    face_detector = FaceDetector(
        model_path='/home/gulingna/OpenFace3/openface3_models/Alignment_RetinaFace.pth',
        device=device
    )

    # LandmarkDetector: uses device_ids (now [0] because only 1 GPU is visible)
    landmark_detector = LandmarkDetector(
        model_path='/home/gulingna/OpenFace3/openface3_models/Landmark_68.pkl',
        device=device,
        device_ids=device_ids
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            break
        process_one(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default='/CIL_PROJECTS/AU_dataset/DISFA/raw_frames')
    parser.add_argument('-d', '--dst', default='/CIL_PROJECTS/AU_dataset/DISFA/aligned_images')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output_size', type=int, default=256)
    parser.add_argument('--transform_size', type=int, default=1024)
    parser.add_argument('--no_padding', action='store_false')

    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    task_queue = mp.Queue(maxsize=2000)
    processes = []

    # Use a free GPU (in your case GPU 1)
    gpu_id = 0
    p = mp.Process(target=worker_loop, args=(task_queue, gpu_id))
    p.start()
    processes.append(p)

    try:
        # Loop over video folders: LeftVideoSN001, LeftVideoSN002, etc.
        for video_dir in sorted(os.listdir(args.src)):
            full_video_path = os.path.join(args.src, video_dir)

            if not os.path.isdir(full_video_path):
                continue

            # create output dir
            dst_video_dir = os.path.join(args.dst, video_dir)
            os.makedirs(dst_video_dir, exist_ok=True)

            # Loop over frames
            frames = sorted(os.listdir(full_video_path))
            frames.reverse()
            for img_name in frames:
                if not img_name.lower().endswith(".jpg"):
                    continue

                raw_img_path = os.path.join(full_video_path, img_name)
                aligned_path = os.path.join(dst_video_dir, img_name)
                if os.path.exists(os.path.join(dst_video_dir, img_name)):
                    continue

                task_queue.put((
                    raw_img_path,
                    aligned_path,
                    args.output_size,
                    args.transform_size,
                    args.no_padding
                ))

    finally:
        # Shutdown worker cleanly
        for _ in processes:
            task_queue.put("STOP")

        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
