"""
NeuroChestYOLO Local Inference & Memory Benchmark Script
Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026

Measures component-level latency (ms), end-to-end throughput,
and memory utilization (RAM in MB) on real radiographs.
"""

import os
import sys
import time
import glob
import json
import torch
import psutil

# Ensure repository root is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model_loader import load_all_models
from backend.utils import preprocess_image, numpy_to_b64, pil_to_numpy
from backend.densenet_inference import CONDITION_COLORS_RGB, overlay_and_contour
from backend.yolo_inference import run_yolo_inference
from backend.risk_scorer import get_risk_level, get_case_flags
from backend.response_builder import build_unified_response

def run_benchmark():
    process = psutil.Process()
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)
    print("=" * 70)
    print("   NEUROCHESTYOLO — COMPREHENSIVE PERFORMANCE BENCHMARK")
    print(f"   Author: Debashis Mohapatra (2026)")
    print(f"   Hardware Device: {'CUDA / GPU' if torch.cuda.is_available() else 'CPU (Single-Thread Concurrency)'}")
    print(f"   Initial Baseline Memory: {ram_initial_mb:.2f} MB")
    print("=" * 70)

    # 1. Load Model Registry
    t0 = time.perf_counter()
    models = load_all_models()
    load_duration_s = time.perf_counter() - t0
    ram_after_load_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n[Models Loaded in {load_duration_s:.2f}s | Current RAM: {ram_after_load_mb:.2f} MB]")

    # Extract stored validation metrics
    nih_checkpoint_path = os.path.join(ROOT_DIR, "models", "nih-model.pth")
    metrics_json_path = os.path.join(ROOT_DIR, "models", "thresholds_and_metrics.json")

    val_loss = None
    mean_auc = None
    if os.path.exists(nih_checkpoint_path):
        ckpt = torch.load(nih_checkpoint_path, map_location="cpu", weights_only=False)
        val_loss = ckpt.get("val_loss")
        mean_auc = ckpt.get("val_mean_auc")

    # 2. Collect sample radiographs
    sample_images = glob.glob(os.path.join(ROOT_DIR, "frontend", "public", "testing_images", "*.png"))
    if not sample_images:
        print("[!] No sample images found in frontend/public/testing_images/")
        return

    print(f"\nBenchmarking inference across {len(sample_images)} clinical chest radiographs...")
    print("-" * 70)

    densenet_times = []
    cam_times = []
    yolo_times = []
    response_times = []
    total_times = []
    peak_ram_mb = ram_after_load_mb

    # Warmup pass
    with open(sample_images[0], "rb") as f:
        warmup_bytes = f.read()
    _t, _p = preprocess_image(warmup_bytes, max_dim=224)
    with torch.inference_mode():
        _ = models["densenet"](_t.to(next(models["densenet"].parameters()).device))

    # Benchmark passes
    for idx, img_path in enumerate(sample_images):
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        t_start = time.perf_counter()

        # Step A: Preprocess
        img_tensor, original_img_pil = preprocess_image(img_bytes, max_dim=224)
        device = next(models["densenet"].parameters()).device
        img_tensor = img_tensor.to(device)

        # Step B: DenseNet-121 Feature Extraction & Forward Pass
        features = None
        def hook(module, input, output):
            nonlocal features
            features = output
        handle = models["densenet"].backbone.norm5.register_forward_hook(hook)

        t_dense_0 = time.perf_counter()
        with torch.inference_mode():
            logits = models["densenet"](img_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        handle.remove()
        t_dense_ms = (time.perf_counter() - t_dense_0) * 1000

        # Step C: Forward Linear CAM Generation
        t_cam_0 = time.perf_counter()
        valid_conditions = []
        for i, label in enumerate(models["label_columns"]):
            if probs[i] >= models["prob_thresholds"].get(label, 0.5) and label != "No Finding":
                valid_conditions.append(label)

        heatmaps = {}
        primary_heatmap_np = None
        if valid_conditions:
            top_label = max(valid_conditions, key=lambda c: probs[models["label_columns"].index(c)])
            class_idx = models["label_columns"].index(top_label)
            with torch.inference_mode():
                weights = models["densenet"].classifier.weight[class_idx][:1024]
                feat_map = features[0]
                cam = torch.einsum('c,chw->hw', weights, feat_map)
                cam = torch.relu(cam)
                if cam.max() > 0:
                    cam = cam / cam.max()
                cam_np = cam.cpu().numpy()

            original_np = pil_to_numpy(original_img_pil)
            visual_np = overlay_and_contour(
                original_np.copy(),
                cam_np,
                models["cam_thresholds"].get(top_label, 0.5),
                CONDITION_COLORS_RGB.get(top_label, (255, 255, 255)),
                top_label
            )
            heatmaps[top_label] = numpy_to_b64(visual_np, quality=75)
            primary_heatmap_np = visual_np

        t_cam_ms = (time.perf_counter() - t_cam_0) * 1000

        # Step D: YOLOv8 Detection
        t_yolo_0 = time.perf_counter()
        yolo_results = run_yolo_inference(
            models["yolo"],
            original_img_pil,
            valid_conditions,
            models["id_to_class"],
            models["overlap_classes"],
            models["yolo_conf"],
            models["yolo_iou"],
            imgsz=224
        )
        t_yolo_ms = (time.perf_counter() - t_yolo_0) * 1000

        # Step E: Response Assembly & Packaging
        t_resp_0 = time.perf_counter()
        risk_level = get_risk_level(valid_conditions)
        case_flags = get_case_flags(valid_conditions, yolo_results["kept_boxes"], models["overlap_classes"])
        densenet_results = {
            "all_results": {c: {"prob": float(probs[i]), "positive": c in valid_conditions} for i, c in enumerate(models["label_columns"])},
            "positive_conditions": valid_conditions,
            "heatmaps": heatmaps,
            "primary_heatmap_np": primary_heatmap_np,
            "heatmap_images": {}
        }
        response = build_unified_response(
            densenet_results,
            yolo_results,
            risk_level,
            case_flags,
            models["prob_thresholds"],
            models["overlap_classes"],
            original_img_pil=original_img_pil
        )
        t_resp_ms = (time.perf_counter() - t_resp_0) * 1000
        t_total_ms = (time.perf_counter() - t_start) * 1000

        current_ram_mb = process.memory_info().rss / (1024 * 1024)
        if current_ram_mb > peak_ram_mb:
            peak_ram_mb = current_ram_mb

        densenet_times.append(t_dense_ms)
        cam_times.append(t_cam_ms)
        yolo_times.append(t_yolo_ms)
        response_times.append(t_resp_ms)
        total_times.append(t_total_ms)

        base_name = os.path.basename(img_path)
        print(f"[{idx+1:02d}/{len(sample_images)}] {base_name:12s} | DenseNet: {t_dense_ms:5.1f}ms | CAM: {t_cam_ms:5.1f}ms | YOLO: {t_yolo_ms:5.1f}ms | Total: {t_total_ms:6.1f}ms | Risk: {risk_level:8s}")

    # 3. Output Aggregated Metrics
    avg_dense = sum(densenet_times) / len(densenet_times)
    avg_cam = sum(cam_times) / len(cam_times)
    avg_yolo = sum(yolo_times) / len(yolo_times)
    avg_resp = sum(response_times) / len(response_times)
    avg_total = sum(total_times) / len(total_times)
    throughput_fps = 1000.0 / avg_total

    print("\n" + "=" * 70)
    print("   EXACT BENCHMARK SUMMARY (READY FOR RESUME & PORTFOLIO)")
    print("=" * 70)
    print(f"• DenseNet-121 Forward Pass Latency:    {avg_dense:.2f} ms")
    print(f"• Forward Linear CAM Generation Latency: {avg_cam:.2f} ms")
    print(f"• YOLOv8 Detection Latency:              {avg_yolo:.2f} ms")
    print(f"• Response Assembly & JPEG Encoding:     {avg_resp:.2f} ms")
    print(f"• Mean End-to-End Latency per Radiograph:{avg_total:.2f} ms ({avg_total/1000:.3f} s)")
    print(f"• Inference Throughput:                  {throughput_fps:.2f} FPS")
    print(f"• Peak Working Memory (RAM RSS):         {peak_ram_mb:.2f} MB")
    if mean_auc is not None and val_loss is not None:
        print(f"• Validation Multi-Label Mean ROC-AUC:   {mean_auc:.4f} ({mean_auc*100:.2f}%)")
        print(f"• Validation Binary Cross-Entropy Loss:  {val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
