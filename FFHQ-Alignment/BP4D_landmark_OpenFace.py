import os
import cv2
import argparse
import numpy as np
import torch.multiprocessing as mp
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector

mp.set_start_method("spawn", force=True)

def process_one(task):
    raw_img_path, save_path = task
    try:
        img_raw = cv2.imread(raw_img_path)
        cropped_face, dets = face_detector.get_face(raw_img_path)

        if dets is None or len(dets) == 0:
            print("[NO FACE]", raw_img_path)
            return

        landmarks = landmark_detector.detect_landmarks(img_raw, dets)
        if landmarks is None or len(landmarks) == 0:
            print("[NO LMK]", raw_img_path)
            return

        shape = np.array(landmarks)[0]  # (68,2)
        np.save(save_path, shape)

        print("[OK]", save_path)

    except Exception as e:
        print("[ERROR]", raw_img_path, e)


def worker_loop(task_queue, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    device_ids = [0]  # local GPU index inside worker

    print(f"[Worker] device={device}, device_ids={device_ids}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    global face_detector, landmark_detector

    face_detector = FaceDetector(
        model_path="/home/gulingna/OpenFace3/openface3_models/Alignment_RetinaFace.pth",
        device=device
    )

    landmark_detector = LandmarkDetector(
        model_path="/home/gulingna/OpenFace3/openface3_models/Landmark_68.pkl",
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
    parser.add_argument("-s","--src", default="./bp4d_missing")
    parser.add_argument("-o","--out", default="./BP4D_landmarks_OpenFace")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    task_queue = mp.Queue(maxsize=2000)
    gpu_id =0  # use V100 GPU3
    p = mp.Process(target=worker_loop, args=(task_queue, gpu_id))
    p.start()

    # traverse aligned images structure: subject / T / frames
    for subject in sorted(os.listdir(args.src)):
        if not subject.startswith(("F","M")):
            continue

        sub_src = os.path.join(args.src, subject)
        sub_dst = os.path.join(args.out, subject)
        os.makedirs(sub_dst, exist_ok=True)

        for vid in sorted(os.listdir(sub_src)):
            vid_src = os.path.join(sub_src, vid)
            vid_dst = os.path.join(sub_dst, vid)
            os.makedirs(vid_dst, exist_ok=True)

            for img_name in sorted(os.listdir(vid_src)):
                if not img_name.lower().endswith(".jpg"):
                    continue

                img_path = os.path.join(vid_src, img_name)
                save_path = os.path.join(vid_dst, img_name.replace(".jpg",".npy"))

                task_queue.put((img_path, save_path))

    task_queue.put("STOP")
    p.join()


if __name__ == "__main__":
    main()
