import os
import gc
import torch
import cv2
import numpy as np
from .utils import pil_to_numpy, numpy_to_b64, normalize_label
from .densenet_inference import CONDITION_COLORS_RGB

def draw_boxes_on_image(image_rgb, kept_boxes):
    """Draw bounding boxes and labels directly onto an RGB image."""
    canvas = image_rgb.copy()
    for box in kept_boxes:
        name = box.get('class_name', box.get('class', ''))
        conf_score = box['conf']
        x1, y1, x2, y2 = box['xyxy']

        color_key = name.replace(' ', '_')
        if color_key not in CONDITION_COLORS_RGB:
            color_key = name
        color_rgb = CONDITION_COLORS_RGB.get(color_key, (255, 255, 255))

        # Draw box in RGB
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_rgb, 3)

        # Draw label in RGB
        label_text = f"YOLO: {name} {conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x1, max(y1-th-10, 0)), (x1+tw+4, y1), color_rgb, -1)
        cv2.putText(canvas, label_text, (x1+2, max(y1-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return canvas

def run_yolo_inference(yolo_model, original_img_pil, positive_conditions, id_to_class, overlap_classes, conf, iou, imgsz=None):
    """Run YOLO detection filtered by NIH classification results (Section 8)."""
    
    # 1. Determine image size: strict 224 for fast low-latency container environments
    target_imgsz = 224

    # 2. Determine eligible classes for detection
    normalized_positive = {normalize_label(c) for c in positive_conditions}
    eligible = normalized_positive & overlap_classes

    original_np = pil_to_numpy(original_img_pil)
    img_display = original_np.copy()
    kept_boxes = []

    if not eligible:
        # Return empty results if no overlap classes were confirmed by NIH
        return {
            "boxes_image": numpy_to_b64(img_display),
            "kept_boxes": [],
            "boxes_image_np": img_display
        }

    # 3. Run YOLO Inference strictly under torch.inference_mode()
    with torch.inference_mode():
        preds = yolo_model.predict(
            source=original_img_pil, 
            imgsz=224, 
            conf=0.25, 
            max_det=3,
            device="cpu",
            verbose=False
        )
        results = preds[0]

        # 4. Filter Boxes: extract native Python coordinate dicts
        for box in results.boxes:
            cid = int(box.cls)
            name = id_to_class.get(str(cid))
            if not name or name not in eligible:
                continue
            conf_score = float(box.conf)
                
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            kept_boxes.append({
                'class': name,
                'class_name': name, 
                'conf': round(conf_score, 4), 
                'xyxy': [x1, y1, x2, y2]
            })

        # Immediately invoke del results and gc.collect()
        del preds
        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 5. Draw boxes onto clean display canvas
    img_display = draw_boxes_on_image(original_np, kept_boxes)

    return {
        "boxes_image": numpy_to_b64(img_display),
        "kept_boxes": kept_boxes,
        "boxes_image_np": img_display
    }
