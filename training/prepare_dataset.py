"""
Dataset Extraction and Formatter for NIH-14 Subset.
Unpacks Parquet partitions, extracts raw radiographs into data/images/,
and constructs standardized multi-label train.csv and val.csv.

Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026
"""

import os
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
NIH_15_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
]

# Mapping from Parquet class integer to canonical NIH-14 label
PARQUET_CLASS_MAP = {
    0: "No Finding",
    1: "Atelectasis",
    2: "Cardiomegaly",
    3: "Effusion",
    4: "Infiltration",
    5: "Mass",
    6: "Nodule",
    7: "Pneumonia",
    8: "Pneumothorax",
    9: "Consolidation",
    10: "Edema",
    11: "Emphysema",
    12: "Fibrosis",
    13: "Pleural_Thickening",
    14: "Hernia"
}

def extract_partition(parquet_path, output_images_dir, split_name, max_samples=None):
    """
    Reads a Parquet partition, saves image bytes to output_images_dir,
    and returns a DataFrame formatted with image_path and 15 binary targets.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    if max_samples and max_samples < len(df):
        df = df.iloc[:max_samples]

    records = []
    print(f"Extracting {len(df)} images for {split_name} split...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Unpacking {split_name}"):
        img_dict = row["image"]
        img_bytes = img_dict.get("bytes")
        img_rel_path = img_dict.get("path") or f"{split_name}_{idx:06d}.png"
        img_filename = os.path.basename(img_rel_path)

        # Save to disk
        out_filepath = os.path.join(output_images_dir, img_filename)
        if not os.path.exists(out_filepath):
            with open(out_filepath, "wb") as f:
                f.write(img_bytes)

        # Build multi-hot target record
        labels_list = row["labels"]
        record = {"image_path": img_filename}
        for cls_name in NIH_15_CLASSES:
            record[cls_name] = 0.0

        for lbl_idx in labels_list:
            mapped_name = PARQUET_CLASS_MAP.get(lbl_idx)
            if mapped_name and mapped_name in record:
                record[mapped_name] = 1.0

        records.append(record)

    out_df = pd.DataFrame(records)
    return out_df

def main():
    import glob
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--g-ronimo--NIH-Chest-X-ray-dataset_10k/snapshots/*")
    snapshots = glob.glob(cache_base)
    if not snapshots:
        raise FileNotFoundError("Hugging Face snapshot cache not found for 10k dataset.")

    snapshot_dir = snapshots[0]
    test_parquet = os.path.join(snapshot_dir, "data/test-00000-of-00001.parquet")
    train_parquet = os.path.join(snapshot_dir, "data/train-00000-of-00001.parquet")

    output_images_dir = os.path.abspath("data/images")
    os.makedirs("data", exist_ok=True)

    # If train_parquet exists, use 80/20 train/val, or split test_parquet
    if os.path.exists(train_parquet):
        # Extract full 10,000 balanced dataset (7,500 train + 2,500 val)
        train_df = extract_partition(train_parquet, output_images_dir, "train", max_samples=None)
        val_df = extract_partition(test_parquet, output_images_dir, "val", max_samples=None)
    else:
        # Split test_parquet into 80/20 train/val
        full_df = extract_partition(test_parquet, output_images_dir, "all", max_samples=2500)
        train_df = full_df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df = full_df.drop(train_df.index).reset_index(drop=True)

    train_csv_path = os.path.abspath("data/train.csv")
    val_csv_path = os.path.abspath("data/val.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print("\nDataset Preparation Complete:")
    print(f"  Images Directory : {output_images_dir} ({len(os.listdir(output_images_dir))} files)")
    print(f"  Train CSV        : {train_csv_path} ({len(train_df)} samples)")
    print(f"  Validation CSV   : {val_csv_path} ({len(val_df)} samples)")
    print("\nClass representation in train set:")
    for col in NIH_15_CLASSES:
        print(f"  {col:<20}: {int(train_df[col].sum())}")

if __name__ == "__main__":
    main()
