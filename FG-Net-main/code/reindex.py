import os
import pandas as pd
from glob import glob

aligned_root = "../data/DISFA/aligned_landmarks"
raw_label_root = "/CIL_PROJECTS/AU_dataset/DISFA/ActionUnit_Labels"

old_train_csv = "../data/DISFA/labels/2/train.csv"
old_test_csv = "../data/DISFA/labels/2/test.csv"

# Load old CSVs only to determine subject split
train_subjects = set(pd.read_csv(old_train_csv)["image_path"].apply(lambda x: x.split("/")[-2]))
test_subjects = set(pd.read_csv(old_test_csv)["image_path"].apply(lambda x: x.split("/")[-2]))

print("Train subjects:", train_subjects)
print("Test subjects:", test_subjects)

# DISFA AU list
au_list = [1, 2, 4, 6, 9, 12, 25, 26]

def load_au_labels(subject):
    au_files = sorted(glob(os.path.join(raw_label_root, subject, f"{subject}_au*.txt")))
    au_data = {au: {} for au in au_list}

    for file in au_files:
        au_name = int(os.path.basename(file).split("_au")[-1].split(".")[0])
        if au_name not in au_list:
            continue

        with open(file, "r") as f:
            for line in f:
                frame, val = line.strip().split(",")
                val = int(val)
                bin_val = 1 if val >=1 else 0
                au_data[au_name][f"{int(frame):05d}.jpg"] = bin_val

    return au_data

def build_labels_for_subject(subject):
    aligned_path = os.path.join(aligned_root, subject)
    frames = sorted(os.listdir(aligned_path))
    au_labels = load_au_labels(subject)

    rows = []
    for frame in frames:
        frame = frame.replace('npy','jpg')
        labels = [au_labels[au][frame] for au in au_list]
        rows.append([f"DISFA/aligned_images/{subject}/{frame}"] + labels)
    return rows

# Rebuild data
new_train = []
new_test = []

for subject in sorted(os.listdir(aligned_root)):
    if not subject.startswith("SN"):
        continue

    print(f"Processing {subject} ...")

    rows = build_labels_for_subject(subject)

    if subject in train_subjects:
        new_train.extend(rows)
    elif subject in test_subjects:
        new_test.extend(rows)
    else:
        print(f"[WARNING] Subject {subject} is not in old split — skipping")

# Save new CSVs
cols = ["image_path", "au1","au2","au4","au6","au9","au12","au25","au26"]
df_train = pd.DataFrame(new_train, columns=cols)
df_test = pd.DataFrame(new_test, columns=cols)

df_train.to_csv("../data/DISFA/labels/2/reindexed_train_1.csv", index=False)
df_test.to_csv("../data/DISFA/labels/2/reindexed_test_1.csv", index=False)

print("DONE — Fully reconstructed CSVs from raw labels")
