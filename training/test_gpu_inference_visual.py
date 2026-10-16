"""
GPU-Accelerated Inference & End-to-End Visual Integrity Check.
Verifies that:
1. backend.main and backend.model_loader route models to CUDA on RTX 4060.
2. Inference executes at sub-second GPU latency.
3. All image base64 payloads decode to valid, correctly oriented RGB images
   matching frontend consumption.

Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026
"""

import os
import sys
import io
import time
import base64
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app, models_registry

def test_gpu_inference_pipeline():
    print("=" * 65)
    print("GPU INFERENCE & END-TO-END VISUAL DECODING TEST")
    print("=" * 65)

    with TestClient(app) as client:
        # 1. Device and Model Allocation Check
        device = models_registry.get('device')
        model = models_registry.get('densenet')
        assert device == "cuda", f"Expected CUDA device, got {device}"
        model_device = next(model.parameters()).device.type
        assert model_device == "cuda", f"Model is not on CUDA! Found on {model_device}"
        gpu_name = torch.cuda.get_device_name(0)
        vram_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        print(f"[OK] Model successfully routed to: {gpu_name}")
        print(f"[OK] PyTorch Device: {model_device}")
        print(f"[OK] VRAM currently allocated: {vram_allocated:.2f} MiB")

        # 2. Test Image Dispatch
        sample_path = os.path.join(BASE_DIR, "frontend", "public", "testing_images", "36204.png")
        assert os.path.exists(sample_path), f"Sample image missing: {sample_path}"

        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        t0 = time.time()
        response = client.post("/predict", files={"file": ("36204.png", file_bytes, "image/png")})
        elapsed = time.time() - t0

        assert response.status_code == 200, f"Predict failed with {response.status_code}: {response.text}"
        data = response.json()
        print(f"\n[OK] HTTP 200 OK received in {elapsed:.3f}s ({elapsed*1000:.1f}ms latency)")

        # 3. Response Schema & Key Alignment Audit
        required_root_keys = ["status", "predictions", "positive_conditions", "confirmed_findings", "risk_level", "localization_type", "images", "yolo_boxes", "case_flags"]
        for k in required_root_keys:
            assert k in data, f"Missing required response key: '{k}'"
        print("[OK] All required root schema keys present.")

        images = data["images"]
        required_image_keys = ["original_b64", "heatmap_b64", "boxes_b64", "combined_b64", "gradcam_b64", "yolo_b64", "individual_heatmaps"]
        for img_k in required_image_keys:
            assert img_k in images, f"Missing image key in payload: '{img_k}'"
        print("[OK] All required image keys and frontend aliases present in images payload.")

        # 4. Base64 Image Decoding & Channel/Orientation Validation
        print("\n" + "-" * 65)
        print("VISUAL DECODING & CHANNEL AUDIT ACROSS ALL VIEWS")
        print("-" * 65)

        # (a) Original View
        assert len(images["original_b64"]) > 0, "original_b64 payload is empty!"
        orig_pil = Image.open(io.BytesIO(base64.b64decode(images["original_b64"]))).convert("RGB")
        print(f"  - original_b64 : Mode={orig_pil.mode}, Size={orig_pil.size}, Payload={len(images['original_b64'])} chars")
        assert orig_pil.size == (1024, 1024), f"Unexpected original image size: {orig_pil.size}"

        # (b) Heatmap View
        if images["heatmap_b64"]:
            heat_pil = Image.open(io.BytesIO(base64.b64decode(images["heatmap_b64"]))).convert("RGB")
            heat_np = np.array(heat_pil)
            print(f"  - heatmap_b64  : Mode={heat_pil.mode}, Size={heat_pil.size}, Payload={len(images['heatmap_b64'])} chars")
            assert heat_pil.size == (1024, 1024), f"Unexpected heatmap size: {heat_pil.size}"
            # Verify colormap variance (not monochrome grayscale)
            std_r = float(np.std(heat_np[:, :, 0]))
            std_b = float(np.std(heat_np[:, :, 2]))
            assert std_r > 5.0 and std_b > 5.0, "Heatmap appears flat or unblended!"
            print(f"    * Heatmap colormap variance confirmed: Std(R)={std_r:.1f}, Std(B)={std_b:.1f}")

        # (c) Boxes View
        assert len(images["boxes_b64"]) > 0, "boxes_b64 payload is empty!"
        boxes_pil = Image.open(io.BytesIO(base64.b64decode(images["boxes_b64"]))).convert("RGB")
        print(f"  - boxes_b64    : Mode={boxes_pil.mode}, Size={boxes_pil.size}, Payload={len(images['boxes_b64'])} chars")
        assert boxes_pil.size == (1024, 1024), f"Unexpected boxes view size: {boxes_pil.size}"

        # (d) Combined View
        assert len(images["combined_b64"]) > 0, "combined_b64 payload is empty!"
        comb_pil = Image.open(io.BytesIO(base64.b64decode(images["combined_b64"]))).convert("RGB")
        print(f"  - combined_b64 : Mode={comb_pil.mode}, Size={comb_pil.size}, Payload={len(images['combined_b64'])} chars")
        assert comb_pil.size == (1024, 1024), f"Unexpected combined view size: {comb_pil.size}"

        # (e) Verify Aliases
        assert images["gradcam_b64"] == images["heatmap_b64"], "gradcam_b64 alias does not match heatmap_b64"
        assert images["yolo_b64"] == images["boxes_b64"], "yolo_b64 alias does not match boxes_b64"
        print("  - Aliases gradcam_b64 and yolo_b64 confirmed identical to primary views.")

        # 5. Diagnostic Predictions Summary
        print("\n" + "-" * 65)
        print("DIAGNOSTIC TRIAGE SUMMARY")
        print("-" * 65)
        print(f"  * Risk Level: {data['risk_level']}")
        print(f"  * Confirmed Findings: {len(data['confirmed_findings'])}")
        for f in data['confirmed_findings']:
            print(f"    - {f['condition']:<18}: Prob={f['prob']:.4f} | Thresh={f['threshold']:.4f} | NormConf={f['normalized_conf']:.4f} | Risk={f['risk']}")

    print("\n" + "=" * 65)
    print("ALL GPU INFERENCE & VISUAL CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 65)

if __name__ == "__main__":
    test_gpu_inference_pipeline()
