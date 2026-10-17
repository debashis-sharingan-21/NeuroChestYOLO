import gc
from .risk_scorer import RISK_MAP
from .utils import numpy_to_b64, pil_to_numpy, normalize_label
from .yolo_inference import draw_boxes_on_image

def build_unified_response(
    densenet_results, 
    yolo_results, 
    risk_level, 
    case_flags, 
    prob_thresholds, 
    overlap_classes,
    original_img_pil=None,
    original_b64=None
):
    """Assemble final JSON response for frontend with aligned keys, blended views, and GC cleanup."""
    
    # 1. Confirmed Findings (Positive conditions sorted by probability)
    confirmed_findings = []
    for cond in densenet_results.get('positive_conditions', []):
        if cond == "No Finding":
            continue
        
        prob = densenet_results['all_results'][cond]['prob']
        threshold = prob_thresholds.get(cond, 0.5)
        
        # Normalized confidence calculation: (prob - threshold) / (1 - threshold)
        normalized_conf = (prob - threshold) / (1.0 - threshold)
        normalized_conf = max(0.0, min(1.0, normalized_conf)) # Clamp 0-1
        
        confirmed_findings.append({
            "condition": cond,
            "prob": prob,
            "threshold": threshold,
            "normalized_conf": round(normalized_conf, 4),
            "risk": RISK_MAP.get(cond, "LOW")
        })
    
    # Sort by probability descending
    confirmed_findings.sort(key=lambda x: x['prob'], reverse=True)

    # 2. Localization Type Mapping
    localization_type = {}
    for cond in densenet_results.get('positive_conditions', []):
        if cond == "No Finding":
            continue
            
        norm_cond = normalize_label(cond)
        if norm_cond in overlap_classes:
            localization_type[cond] = "boxes_and_heatmap"
        else:
            localization_type[cond] = "heatmap_only"

    # 3. Image Views Generation
    # (a) True Original Image
    if original_b64:
        clean_original_b64 = original_b64
    elif original_img_pil is not None:
        clean_original_b64 = numpy_to_b64(pil_to_numpy(original_img_pil), quality=80)
    else:
        clean_original_b64 = yolo_results.get('boxes_image', '')

    # (b) Heatmap Image (Primary finding)
    heatmaps = densenet_results.get('heatmaps', {})
    primary_heatmap_b64 = list(heatmaps.values())[0] if heatmaps else ""

    # (c) Boxes Image (YOLO detection)
    boxes_b64 = yolo_results.get('boxes_image', clean_original_b64)

    # (d) Combined Image (Heatmap blended with YOLO bounding boxes)
    primary_heatmap_np = densenet_results.get('primary_heatmap_np')
    kept_boxes = yolo_results.get('kept_boxes', [])

    if primary_heatmap_np is not None and len(kept_boxes) > 0:
        # Draw YOLO detection boxes on top of the Grad-CAM activation heatmap
        blended_np = draw_boxes_on_image(primary_heatmap_np, kept_boxes)
        combined_b64 = numpy_to_b64(blended_np, quality=80)
        del blended_np
    elif primary_heatmap_b64:
        combined_b64 = primary_heatmap_b64
    elif len(kept_boxes) > 0:
        combined_b64 = boxes_b64
    else:
        combined_b64 = clean_original_b64

    # 4. Assemble full response
    response = {
        "status": "success",
        "predictions": densenet_results.get('all_results', {}),
        "positive_conditions": densenet_results.get('positive_conditions', []),
        "confirmed_findings": confirmed_findings,
        "risk_level": risk_level,
        "localization_type": localization_type,
        "images": {
            "original_b64": clean_original_b64,
            "heatmap_b64": primary_heatmap_b64,
            "boxes_b64": boxes_b64,
            "combined_b64": combined_b64
        },
        "yolo_boxes": kept_boxes,
        "case_flags": case_flags,
        "report": None
    }
    
    # 5. Clean up allocated image buffers and invoke garbage collection
    gc.collect()

    return response
