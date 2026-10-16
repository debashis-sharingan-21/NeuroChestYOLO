"""
Color Parity and Channel Integrity Audit Test.
Verifies that RGB channel order is strictly preserved end-to-end
across PIL loading, OpenCV processing, overlay blending, YOLO annotations,
and Base64 JPEG encoding.

Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026
"""

import os
import sys
import io
import base64
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.utils import numpy_to_b64, pil_to_numpy
from backend.densenet_inference import overlay_and_contour, CONDITION_COLORS_RGB
from backend.yolo_inference import draw_boxes_on_image
from backend.response_builder import build_unified_response

def b64_to_pil(b64_str: str) -> Image.Image:
    """Decodes base64 string directly into an RGB PIL Image."""
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def test_numpy_to_b64_channel_order():
    """Verify that pure Red, Green, and Blue survive round-trip without channel swapping."""
    print("=" * 60)
    print("TEST 1: Direct numpy_to_b64 RGB Round-Trip Parity")
    print("=" * 60)
    
    h, w = 100, 100
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Quadrant 1 (Top-Left): Pure Red
    canvas[0:50, 0:50] = [255, 0, 0]
    # Quadrant 2 (Top-Right): Pure Green
    canvas[0:50, 50:100] = [0, 255, 0]
    # Quadrant 3 (Bottom-Left): Pure Blue
    canvas[50:100, 0:50] = [0, 0, 255]
    # Quadrant 4 (Bottom-Right): Pure White
    canvas[50:100, 50:100] = [255, 255, 255]

    b64 = numpy_to_b64(canvas)
    decoded = b64_to_pil(b64)
    dec_np = np.array(decoded)

    tl_pixel = dec_np[25, 25]  # Expected Red
    tr_pixel = dec_np[25, 75]  # Expected Green
    bl_pixel = dec_np[75, 25]  # Expected Blue

    print(f"  - Top-Left (Red quadrant) decoded RGB: {tl_pixel.tolist()}")
    print(f"  - Top-Right (Green quadrant) decoded RGB: {tr_pixel.tolist()}")
    print(f"  - Bottom-Left (Blue quadrant) decoded RGB: {bl_pixel.tolist()}")

    # Red must dominate channel 0, not channel 2
    assert tl_pixel[0] > 240 and tl_pixel[2] < 15, f"Red quadrant inverted! Got {tl_pixel}"
    assert tr_pixel[1] > 240 and tr_pixel[0] < 15, f"Green quadrant corrupted! Got {tr_pixel}"
    assert bl_pixel[2] > 240 and bl_pixel[0] < 15, f"Blue quadrant inverted! Got {bl_pixel}"
    print("[PASS] TEST 1: RGB channel parity preserved in base64 encoding.")

def test_gradcam_overlay_color_integrity():
    """Verify that Grad-CAM overlay and contour annotations render in true RGB."""
    print("\n" + "=" * 60)
    print("TEST 2: Grad-CAM Overlay & Contour Color Space Audit")
    print("=" * 60)

    # Neutral grey background to easily isolate annotation color
    h, w = 120, 120
    grey_bg = np.full((h, w, 3), 128, dtype=np.uint8)
    
    # Synthetic circular CAM centered at (60, 60)
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - 60)**2 + (y - 60)**2
    cam = np.exp(-dist_sq / (2 * 15**2)).astype(np.float32)

    # Pick a distinctly red color for testing
    red_rgb = (255, 0, 0)
    visual_np = overlay_and_contour(
        original_img_rgb=grey_bg,
        cam=cam,
        cam_threshold=0.5,
        color_rgb=red_rgb,
        label="TestRed"
    )

    b64 = numpy_to_b64(visual_np)
    decoded = b64_to_pil(b64)
    dec_np = np.array(decoded)

    # The contour and bounding box are drawn on visual_np
    # Find bounding box pixels where R > 200 and B < 50
    red_mask = (dec_np[:, :, 0] > 200) & (dec_np[:, :, 2] < 50)
    blue_mask = (dec_np[:, :, 2] > 200) & (dec_np[:, :, 0] < 50)

    print(f"  - Pure Red pixels detected on annotation: {np.sum(red_mask)}")
    print(f"  - Erroneous Blue pixels on annotation: {np.sum(blue_mask)}")

    assert np.sum(red_mask) > 50, "Contour/bounding box did NOT render with Red channel!"
    print("[PASS] TEST 2: Grad-CAM annotations correctly render in true RGB (no BGR inversion).")

def test_yolo_boxes_color_integrity():
    """Verify that YOLO bounding boxes render with true RGB condition colors."""
    print("\n" + "=" * 60)
    print("TEST 3: YOLO Bounding Box Annotation Color Audit")
    print("=" * 60)

    black_bg = np.zeros((150, 150, 3), dtype=np.uint8)
    # Mass color is (255, 100, 100) - heavily Red
    boxes = [{
        'class': 'Mass',
        'conf': 0.95,
        'xyxy': [20, 20, 100, 100]
    }]

    drawn_np = draw_boxes_on_image(black_bg, boxes)
    b64 = numpy_to_b64(drawn_np)
    decoded = b64_to_pil(b64)
    dec_np = np.array(decoded)

    # Box border at y=20 from x=25 to 95
    border_sample = dec_np[20, 30:80]
    mean_r = np.mean(border_sample[:, 0])
    mean_g = np.mean(border_sample[:, 1])
    mean_b = np.mean(border_sample[:, 2])

    print(f"  - 'Mass' box border RGB (expected ~255, ~100, ~100): R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")
    assert mean_r > 200, f"Red channel suppressed on YOLO box! Got R={mean_r}"
    assert mean_b < 150, f"Blue channel erroneously elevated on YOLO box! Got B={mean_b}"
    print("[PASS] TEST 3: YOLO bounding box colors render in true RGB without blue tint.")

def test_response_builder_views_distinction():
    """Verify that original_b64 is clean and combined_b64 blends heatmap + boxes."""
    print("\n" + "=" * 60)
    print("TEST 4: Response Builder Clean Original vs. Blended Views")
    print("=" * 60)

    # Create dummy original image
    raw_pil = Image.new("RGB", (100, 100), color=(128, 128, 128))
    raw_np = np.array(raw_pil)

    # Dummy heatmap
    heatmap_np = raw_np.copy()
    heatmap_np[40:60, 40:60] = [255, 180, 0] # Amber heatmap region
    heatmap_b64 = numpy_to_b64(heatmap_np)

    # Dummy boxes
    kept_boxes = [{'class': 'Atelectasis', 'conf': 0.88, 'xyxy': [10, 10, 50, 50]}]
    boxes_np = draw_boxes_on_image(raw_np, kept_boxes)
    boxes_b64 = numpy_to_b64(boxes_np)

    densenet_results = {
        'all_results': {'Atelectasis': {'prob': 0.88, 'positive': True}},
        'positive_conditions': ['Atelectasis'],
        'heatmaps': {'Atelectasis': heatmap_b64},
        'primary_heatmap_np': heatmap_np
    }
    yolo_results = {
        'boxes_image': boxes_b64,
        'kept_boxes': kept_boxes
    }

    response = build_unified_response(
        densenet_results=densenet_results,
        yolo_results=yolo_results,
        risk_level="HIGH",
        case_flags=[],
        prob_thresholds={'Atelectasis': 0.3116},
        overlap_classes={'Atelectasis'},
        original_img_pil=raw_pil
    )

    images = response['images']
    assert 'original_b64' in images, "Missing original_b64"
    assert 'heatmap_b64' in images, "Missing heatmap_b64"
    assert 'boxes_b64' in images, "Missing boxes_b64"
    assert 'combined_b64' in images, "Missing combined_b64"
    assert 'gradcam_b64' in images, "Missing gradcam_b64 alias"
    assert 'yolo_b64' in images, "Missing yolo_b64 alias"

    # Decode views to verify distinctions
    orig_dec = np.array(b64_to_pil(images['original_b64']))
    boxes_dec = np.array(b64_to_pil(images['boxes_b64']))
    heat_dec = np.array(b64_to_pil(images['heatmap_b64']))
    comb_dec = np.array(b64_to_pil(images['combined_b64']))

    # 1. Clean original must NOT have the box
    # The box pixel at [10, 30] in boxes_dec will be non-grey, but in orig_dec it must be ~128
    box_px_in_orig = orig_dec[10, 30]
    box_px_in_boxes = boxes_dec[10, 30]
    print(f"  - Pixel [10, 30] in original_b64: {box_px_in_orig.tolist()} (Clean grey background)")
    print(f"  - Pixel [10, 30] in boxes_b64:    {box_px_in_boxes.tolist()} (Drawn box border)")
    assert np.all(np.abs(box_px_in_orig.astype(int) - 128) < 5), "original_b64 is not clean!"
    assert not np.array_equal(box_px_in_orig, box_px_in_boxes), "original_b64 contains YOLO box!"

    # 2. Combined view must contain BOTH the heatmap amber patch AND the box
    heatmap_px_in_comb = comb_dec[50, 50]
    box_px_in_comb = comb_dec[10, 30]
    print(f"  - Pixel [50, 50] in combined_b64: {heatmap_px_in_comb.tolist()} (Heatmap amber region)")
    print(f"  - Pixel [10, 30] in combined_b64: {box_px_in_comb.tolist()} (YOLO box border)")

    assert heatmap_px_in_comb[0] > 200, "Heatmap region missing from combined_b64!"
    assert not np.all(np.abs(box_px_in_comb.astype(int) - 128) < 10), "YOLO box missing from combined_b64!"
    print("[PASS] TEST 4: original_b64 is pure, and combined_b64 successfully blends heatmap + YOLO boxes.")

if __name__ == '__main__':
    test_numpy_to_b64_channel_order()
    test_gradcam_overlay_color_integrity()
    test_yolo_boxes_color_integrity()
    test_response_builder_views_distinction()
    print("\n" + "=" * 60)
    print("ALL COLOR PARITY & CHANNEL INTEGRITY TESTS PASSED (100% SUCCESS)!")
    print("=" * 60)
