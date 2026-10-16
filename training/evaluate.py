"""
Model Evaluation, Youden Threshold Calibration & Metric Exporter.
Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, precision_score, recall_score, f1_score

from .dataset import CXRDataset, NIH_15_CLASSES
from .model import DenseNetCustom

DEFAULT_CAM_THRESHOLDS = {
    "Pneumonia": 0.40,
    "Edema": 0.40,
    "Infiltration": 0.42,
    "Consolidation": 0.42,
    "Nodule": 0.60,
    "Mass": 0.58,
    "Pneumothorax": 0.62,
    "Emphysema": 0.52,
    "Cardiomegaly": 0.33,
    "Atelectasis": 0.47,
    "Effusion": 0.43,
    "Fibrosis": 0.50,
    "Pleural_Thickening": 0.50,
    "Hernia": 0.45,
    "No Finding": 0.50
}

def find_thresholds_youden(probs, targets, label_cols, min_precision=0.10):
    """
    Computes optimal clinical decision thresholds using Youden's J statistic
    with a minimum precision floor fallback.
    """
    thresholds = {}
    diagnostics = {}

    for i, label in enumerate(label_cols):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        if len(np.unique(y_true)) < 2 or y_true.sum() < 5:
            thresholds[label] = 0.50
            diagnostics[label] = {"method": "fallback", "note": "Insufficient positive cases"}
            continue

        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        best_j_idx = np.argmax(tpr - fpr)
        youden_thresh = float(roc_thresh[best_j_idx])

        prec_j = precision_score(y_true, (y_prob >= youden_thresh).astype(int), zero_division=0)
        if prec_j >= min_precision:
            thresholds[label] = round(youden_thresh, 4)
            diagnostics[label] = {
                "method": "youden_j",
                "prec": round(float(prec_j), 4),
                "tpr": round(float(tpr[best_j_idx]), 4),
                "fpr": round(float(fpr[best_j_idx]), 4)
            }
        else:
            precision_arr, recall_arr, pr_thresh = precision_recall_curve(y_true, y_prob)
            p, r = precision_arr[:-1], recall_arr[:-1]
            valid = p >= min_precision
            if valid.any():
                best_idx = np.where(valid, r, -1).argmax()
                thresholds[label] = round(float(pr_thresh[best_idx]), 4)
                diagnostics[label] = {
                    "method": "precision_floor",
                    "prec": round(float(p[best_idx]), 4),
                    "rec": round(float(r[best_idx]), 4)
                }
            else:
                with np.errstate(invalid="ignore"):
                    f1_vals = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
                thresholds[label] = round(float(pr_thresh[np.argmax(f1_vals)]), 4)
                diagnostics[label] = {"method": "f1_fallback", "note": "Precision floor unreachable"}

    # Apply clinical overrides for weak/ambiguous conditions
    manual_overrides = {
        "Infiltration": 0.75,
        "Nodule": 0.80,
        "Pneumonia": 0.87,
        "No Finding": 0.50
    }
    for k, v in manual_overrides.items():
        if k in thresholds:
            thresholds[k] = v

    return thresholds, diagnostics


def evaluate_and_export(model_path, test_csv, images_dir, output_json="models/thresholds_and_metrics.json", device="cpu"):
    """
    Evaluates a trained model checkpoint on test data, finds optimal thresholds,
    and exports thresholds_and_metrics.json for runtime inference.
    """
    checkpoint = torch.load(model_path, map_location=device)
    label_cols = checkpoint.get("label_columns", NIH_15_CLASSES)

    model = DenseNetCustom(num_classes=len(label_cols), pretrained=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    dataset = CXRDataset(test_csv, images_dir, label_cols=label_cols, is_train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)

    thresholds, diagnostics = find_thresholds_youden(probs, targets, label_cols)

    export_payload = {
        "probability_thresholds": thresholds,
        "cam_thresholds": DEFAULT_CAM_THRESHOLDS,
        "diagnostics": diagnostics,
        "author": "Debashis Mohapatra",
        "dataset_evaluated": test_csv
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"Calibration successful! Exported config to {output_json}")
    return export_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate & Calibrate CXR Thresholds")
    parser.add_argument("--model", type=str, default="models/nih-model.pth")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/thresholds_and_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_and_export(args.model, args.csv, args.images_dir, args.output, device=args.device)
