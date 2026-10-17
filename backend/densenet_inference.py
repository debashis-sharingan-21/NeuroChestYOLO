import os
import gc
import torch
import torch.nn as nn
import numpy as np
import cv2
from .utils import pil_to_numpy, numpy_to_b64

# Per-condition colors as defined in Section 7
CONDITION_COLORS_RGB = {
    'Atelectasis':       (255, 200,  80),
    'Cardiomegaly':      ( 80, 255, 160),
    'Effusion':          ( 80, 160, 255),
    'Infiltration':      (200, 130, 255),
    'Mass':              (255, 100, 100),
    'Nodule':            (255,  80, 200),
    'Pneumonia':         (255,  80,  80),
    'Pneumothorax':      ( 80, 220, 255),
    'Consolidation':     (255, 180,  80),
    'Edema':             (160, 255,  80),
    'Emphysema':         (255, 255,  80),
    'Fibrosis':          (180, 140, 255),
    'Pleural_Thickening':(100, 200, 200),
    'Hernia':            (200, 200, 100),
    'No Finding':        (200, 200, 200)
}

def run_densenet_inference(model, img_tensor, original_img_pil, label_cols, prob_thresholds, cam_thresholds):
    """Run DenseNet inference with memory optimization for constrained environments."""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # 1. Standard Prediction strictly under torch.inference_mode()
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    # 2. Process Results
    results = {}
    positive_conditions = []
    for i, label in enumerate(label_cols):
        prob = float(probs[i])
        is_positive = prob >= prob_thresholds.get(label, 0.5)
        
        if is_positive:
            positive_conditions.append(label)
            
        results[label] = {
            "prob": round(prob, 4),
            "positive": is_positive
        }
    
    # 3. GradCAM & Visualization: restricted to single top positive condition
    original_np = pil_to_numpy(original_img_pil)
    heatmaps = {}
    primary_heatmap_np = None
    heatmap_images = {}
    
    valid_conditions = [c for c in positive_conditions if c != "No Finding"]
    
    # Only hook gradients and compute backprop if at least one positive finding exists
    if valid_conditions:
        # Select single top positive condition by highest probability
        top_label = max(valid_conditions, key=lambda c: results[c]["prob"])
        class_idx = label_cols.index(top_label)
        target_layer = model.backbone.norm5
        
        cam = compute_gradcam(model, img_tensor, class_idx, target_layer)
        
        color_rgb = CONDITION_COLORS_RGB.get(top_label, (255, 255, 255))
        cam_threshold = cam_thresholds.get(top_label, 0.5)
        
        visual_np = overlay_and_contour(
            original_np.copy(), 
            cam, 
            cam_threshold, 
            color_rgb, 
            top_label
        )
        
        heatmaps[top_label] = numpy_to_b64(visual_np, quality=80)
        heatmap_images[top_label] = visual_np
        primary_heatmap_np = visual_np
        
        # Explicit cleanup of cam array
        del cam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return {
        "all_results": results,
        "positive_conditions": positive_conditions,
        "heatmaps": heatmaps,
        "primary_heatmap_np": primary_heatmap_np,
        "heatmap_images": heatmap_images
    }

def compute_gradcam(model, img_tensor, class_idx, target_layer):
    """Compute normalized GradCAM mask with targeted autograd and immediate memory cleanup."""
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    fh = target_layer.register_forward_hook(forward_hook)

    device = next(model.parameters()).device
    input_tensor = img_tensor.to(device)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output = model(input_tensor)
        score = output[0, class_idx]
        act = activations[0]
        # Targeted backward gradients using torch.autograd.grad with retain_graph=False
        grads = torch.autograd.grad(score, act, retain_graph=False)[0]

    fh.remove()

    # Weight activations by mean gradients
    weights = grads[0].mean(dim=(1, 2))  # [1024]
    cam = (weights[:, None, None] * act[0]).sum(dim=0)  # [H, W]

    # ReLU on CAM
    cam = torch.relu(cam)

    # Normalize
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    cam_np = cam.detach().cpu().numpy()

    # Immediately delete tensors to reclaim memory
    del activations, grads, act, weights, cam, output, score
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return cam_np

def overlay_and_contour(original_img_rgb, cam, cam_threshold, color_rgb, label):
    """Draw GradCAM heatmap and bounding contour on image (Section 7)."""
    h, w = original_img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Heatmap overlay (Jet colormap converted to RGB)
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img_rgb, 0.55, heatmap_rgb, 0.45, 0)
    del heatmap, heatmap_rgb

    # 2. Contour extraction at cam_threshold
    thresh_val = int(cam_threshold * 255)
    _, binary = cv2.threshold(
        np.uint8(255 * cam_resized), thresh_val, 255, cv2.THRESH_BINARY
    )
    del cam_resized
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    del binary

    if contours:
        # Pick the largest contour by area
        best_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [best_cnt], -1, color_rgb, 2)
        
        # Draw bounding box and label
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color_rgb, 2)
        
        # Label background
        label_text = f"CAM: {label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(overlay, (x, max(y-th-10, 0)), (x+tw+4, y), color_rgb, -1)
        cv2.putText(overlay, label_text, (x+2, max(y-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    return overlay
