import os
import argparse
import numpy as np
import scipy.ndimage
import PIL.Image
import face_alignment
from tqdm import tqdm

# =========================================================
# FFHQ-style face alignment function
# =========================================================
def image_align(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=False):
    lm = np.array(face_landmarks)
    lm_eye_left      = lm[36 : 42, :2]
    lm_eye_right     = lm[42 : 48, :2]
    lm_mouth_outer   = lm[48 : 60, :2]

    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    if not os.path.isfile(src_file):
        print(f"[Warn] Cannot find image {src_file}")
        return

    img = PIL.Image.open(src_file)

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))),
            int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))),
           int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))

    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3])
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size),
                        PIL.Image.QUAD, (quad + 0.5).flatten(),
                        PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    img.save(dst_file, "png")


def os.environ["OPENFACE_LOGDIR"] = os.path.expanduser("~/openface_logs")
face_detector = FaceDetector(model_path='/home/gulingna/OpenFace3/openface3_models/Alignment_RetinaFace.pth')
landmark_detector = LandmarkDetector(model_path='/home/gulingna/OpenFace3/openface3_models/Landmark_68.pkl')

def openface_landmarks(img_path):
    img_raw = cv2.imread(img_path)
    cropped_face, dets = face_detector.get_face(img_path)

    if dets is None or len(dets) == 0:
        print(f"No face found in {img_path}")
        exit()

    # ----------------------------------------------------
    # 4. Detect landmarks
    # ----------------------------------------------------
    landmarks = landmark_detector.detect_landmarks(img_raw, dets)
    if landmarks is None or len(landmarks) == 0:
        print(f"No landmarks found in {img_path}")
        exit()

    # Get numpy array (shape: [98, 2])
    landmarks = np.array(landmarks)
    return landmarks(img_path):
    img_raw = cv2.imread(img_path)
    cropped_face, dets = face_detector.get_face(img_path)

    if dets is None or len(dets) == 0:
        print(f"No face found in {img_path}")
        exit()

    # ----------------------------------------------------
    # 4. Detect landmarks
    # ----------------------------------------------------
    landmarks = landmark_detector.detect_landmarks(img_raw, dets)
    if landmarks is None or len(landmarks) == 0:
        print(f"No landmarks found in {img_path}")
        exit()

    # Get numpy array (shape: [98, 2])
    landmarks = np.array(landmarks)
    return landmarks


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align BP4D videos recursively.")
    parser.add_argument("-s", "--src", default="/CIL_PROJECTS/AU_dataset/BP4D/videos", help="BP4D raw frames root")
    parser.add_argument("-d", "--dst", default="/CIL_PROJECTS/AU_dataset/BP4D/aligned_images_OpenFace", help="output aligned frames root")
    parser.add_argument("-o", "--output_size", default=256, type=int)
    parser.add_argument("-t", "--transform_size", default=1024, type=int)
    parser.add_argument("--no_padding", action="store_false")
    parser.add_argument("--failed_log", default="bp4d_failed_align_1112.txt", help="log of failed frames")
    args = parser.parse_args()

    # Init face alignment
    landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    failed_frames = []

    # Traverse all .jpg recursively
    for root, _, files in os.walk(args.src):
        for img_name in tqdm(sorted(files), desc=os.path.basename(root)):
            if not img_name.lower().endswith(".jpg"):
                continue
            src_path = os.path.join(root, img_name)

            # Construct target path (mirror structure)
            rel_path = os.path.relpath(src_path, args.src)
            dst_path = os.path.join(args.dst, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            try:
                landmarks = landmarks_detector.get_landmarks(src_path)
            except Exception as e:
                print(f"[Error] {src_path}: {e}")
                failed_frames.append(src_path)
                continue

            if landmarks is None or len(landmarks) == 0:
                failed_frames.append(src_path)
                continue

            try:
                image_align(src_path, dst_path, landmarks[0], args.output_size, args.transform_size)
            except Exception as e:
                print(f"[Align Error] {src_path}: {e}")
                failed_frames.append(src_path)

    # Write failed log
    if failed_frames:
        with open(args.failed_log, "w") as f:
            for path in failed_frames:
                f.write(path + "\n")
        print(f"{len(failed_frames)} frames failed, logged to {args.failed_log}")
    else:
        print("All frames aligned successfully!")