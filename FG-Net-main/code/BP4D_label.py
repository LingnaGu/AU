import os
import pandas as pd
from tqdm import tqdm

OLD_TRAIN = "../data/BP4D/labels/2/train.csv"
OLD_TEST  = "../data/BP4D/labels/2/test.csv"

OCC_DIR   = "/CIL_PROJECTS/AU_dataset/BP4D/AUCoding/AU_OCC"
IMG_DIR   = "../data/BP4D/aligned_landmarks_face_alignment"

OUT_TRAIN = "../data/BP4D/labels/2/bp4d_train_face_alignment.csv"
OUT_TEST  = "../data/BP4D/labels/2/bp4d_test_face_alignment.csv"

# AU indices = column indices in OCC files
TARGET_AUS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]

# =========================
# Failed sample list provided
# =========================
failed = []
'''
# Range F017/T3/200-215
for f in range(200, 216):
    failed.append(("F017", "T3", f))

# Individual fails
failed += [
    ("F017", "T3", 222),
    ("F017", "T3", 198),
    ("F017", "T3", 220),
    ("F017", "T3", 221),
    ("F017", "T3", 217),
    ("F017", "T3", 216),
    ("F017", "T3", 219),
    ("M010", "T4", 1542),
    ("M010", "T4", 1532),
    ("F007", "T5", 1150),
]
failed = set(failed)

'''
def infer_task_sets(old_train_path, old_test_path):
    """Use old csv ONLY to infer which (subject, task) belong to train/test."""
    train_df = pd.read_csv(old_train_path)
    test_df  = pd.read_csv(old_test_path)

    train_tasks = set()
    test_tasks  = set()

    for img in train_df["image_path"]:
        parts = img.split("/")
        subject = parts[-3]  # e.g. F001
        task    = parts[-2]  # e.g. T1
        train_tasks.add((subject, task))

    for img in test_df["image_path"]:
        parts = img.split("/")
        subject = parts[-3]
        task    = parts[-2]
        test_tasks.add((subject, task))

    # Optional: warn if any task appears in both sets
    overlap = train_tasks & test_tasks
    if overlap:
        print("[WARNING] tasks appearing in both train and test:", overlap)

    return train_tasks, test_tasks


def get_padding_length(align_dir):
    """Infer zero-padding length from jpg filenames in aligned_images."""
    files = [f for f in os.listdir(align_dir) if f.lower().endswith(".npy")]
    if not files:
        return None
    stem = os.path.splitext(files[0])[0]
    return len(stem)


def build_split(task_set, out_csv):
    rows = []

    for subject, task in tqdm(sorted(task_set)):
        occ_path = os.path.join(OCC_DIR, f"{subject}_{task}.csv")
        align_dir = os.path.join(IMG_DIR, subject, task)

        if not os.path.exists(occ_path):
            print(f"[MISSING OCC] {occ_path}")
            continue
        if not os.path.isdir(align_dir):
            print(f"[MISSING IMG DIR] {align_dir}")
            continue

        pad_len = get_padding_length(align_dir)
        if pad_len is None:
            print(f"[NO JPG] {align_dir}")
            continue

        df = pd.read_csv(occ_path)
        # Replace 9 (unknown) with 0
        #df = pd.read_csv(occ_path).replace(9, 0)

        for _, r in df.iterrows():
            frame = int(r[0]) -1 

            #if (subject, task, frame) in failed:
            #    continue

            # extract AU labels directly
            #print(df.columns)
            #labels = [int(r[au]) if au in df.columns else 0 for au in TARGET_AUS]
            labels = [int(r.get(str(au), 0)) for au in TARGET_AUS]
            

            # now convert to jpg frame index
            frame_img = frame
            img_name = f"{frame_img:0{pad_len}d}.npy"
            img_path = os.path.join(align_dir, img_name)

            if not os.path.exists(img_path):
                continue
            #print(img_path)
            img_path = img_path.replace('landmarks_face_alignment','images_ori').replace('npy','jpg')
            #print(img_path)
            rows.append([img_path] + labels)


    columns = ["image_path"] + [f"au{a}" for a in TARGET_AUS]
    out_df = pd.DataFrame(rows, columns=columns)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(out_df)} samples")


if __name__ == "__main__":
    train_tasks, test_tasks = infer_task_sets(OLD_TRAIN, OLD_TEST)
    print("Train tasks:", train_tasks)
    print("Test tasks:", test_tasks)

    build_split(train_tasks, OUT_TRAIN)
    build_split(test_tasks, OUT_TEST)
