import sys
import os
import torch
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.model_loader import load_all_models, DenseNetCustom
from backend.densenet_inference import run_densenet_inference, compute_gradcam
from backend.yolo_inference import run_yolo_inference
from backend.risk_scorer import get_risk_level, get_case_flags
from backend.response_builder import build_unified_response
from backend.utils import preprocess_image

def test_backend_inference():
    print("=" * 60)
    print("1. LOADING ALL MODELS VIA backend.model_loader")
    print("=" * 60)
    
    registry = load_all_models(
        nih_pth_path='models/nih-model.pth',
        thresholds_path='models/thresholds_and_metrics.json',
        yolo_pt_path='models/best_chestxdet.pt',
        yolo_config_path='models/yolo_pipeline_config.json'
    )
    
    assert isinstance(registry['densenet'], DenseNetCustom), "densenet must be an instance of DenseNetCustom"
    print(f"[OK] DenseNetCustom loaded successfully.")
    print(f"[OK] Total classes: {len(registry['label_columns'])}")
    print(f"[OK] Classes: {registry['label_columns']}")
    print(f"[OK] Calibrated probability thresholds loaded: {len(registry['prob_thresholds'])} conditions.")
    print(f"[OK] YOLOv8 model loaded: {type(registry['yolo'])}")

    print("\n" + "=" * 60)
    print("2. TESTING FORWARD PASS & PREPROCESSING ON SAMPLE IMAGE")
    print("=" * 60)
    
    sample_img_path = os.path.join(BASE_DIR, 'frontend', 'public', 'testing_images', '36204.png')
    assert os.path.exists(sample_img_path), f"Sample image not found: {sample_img_path}"
    
    with open(sample_img_path, 'rb') as f:
        img_bytes = f.read()
        
    img_tensor, original_pil = preprocess_image(img_bytes)
    print(f"[OK] Image preprocessed. Tensor shape: {img_tensor.shape}, PIL size: {original_pil.size}")
    
    # Direct forward pass to inspect logits and probabilities
    model = registry['densenet']
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        
    assert logits.shape == (1, len(registry['label_columns'])), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"
    assert not torch.isinf(logits).any(), "Logits contain Inf values!"
    assert not torch.isnan(probs).any(), "Probabilities contain NaN values!"
    print(f"[OK] Raw logits shape: {logits.shape}")
    print(f"[OK] Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}] (No NaN/Inf)")
    print(f"[OK] Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    print("\n" + "=" * 60)
    print("3. TESTING DENSENET INFERENCE & GRAD-CAM ACTIVATION MAPS")
    print("=" * 60)
    
    densenet_results = run_densenet_inference(
        model=model,
        img_tensor=img_tensor.clone(),
        original_img_pil=original_pil,
        label_cols=registry['label_columns'],
        prob_thresholds=registry['prob_thresholds'],
        cam_thresholds=registry['cam_thresholds']
    )
    
    print(f"[OK] DenseNet inference completed.")
    print(f"[OK] Positive conditions identified: {densenet_results['positive_conditions']}")
    
    for label, res in densenet_results['all_results'].items():
        thresh = registry['prob_thresholds'][label]
        flag = " [POSITIVE]" if res['positive'] else ""
        print(f"  - {label:<20}: prob = {res['prob']:.4f} (calibrated threshold = {thresh:.4f}){flag}")
        
    # Check Grad-CAM generation for at least one disease class
    # Test Grad-CAM explicitly on target layer
    target_layer = model.backbone.norm5
    test_cam = compute_gradcam(model, img_tensor.clone(), class_idx=0, target_layer=target_layer)
    assert test_cam.shape == (7, 7), f"Expected Grad-CAM shape (7,7), got {test_cam.shape}"
    assert not np.isnan(test_cam).any(), "Grad-CAM contains NaN values!"
    assert test_cam.min() >= 0.0 and test_cam.max() <= 1.0, "Grad-CAM is not normalized in [0, 1]!"
    print(f"[OK] Grad-CAM computed for class 0 ('{registry['label_columns'][0]}'): shape={test_cam.shape}, range=[{test_cam.min():.4f}, {test_cam.max():.4f}]")

    print("\n" + "=" * 60)
    print("4. TESTING YOLO INFERENCE & CONDITIONAL OVERLAYS")
    print("=" * 60)
    
    yolo_results = run_yolo_inference(
        registry['yolo'],
        original_pil,
        densenet_results['positive_conditions'],
        registry['id_to_class'],
        registry['overlap_classes'],
        registry['yolo_conf'],
        registry['yolo_iou'],
        registry['yolo_imgsz']
    )
    print(f"[OK] YOLO inference ran successfully.")
    print(f"[OK] Kept boxes count: {len(yolo_results['kept_boxes'])}")
    print(f"[OK] Boxes image base64 length: {len(yolo_results['boxes_image'])} chars")

    print("\n" + "=" * 60)
    print("5. TESTING RISK SCORER & UNIFIED RESPONSE BUILDER")
    print("=" * 60)
    
    risk_level = get_risk_level(densenet_results['positive_conditions'])
    case_flags = get_case_flags(
        densenet_results['positive_conditions'], 
        yolo_results['kept_boxes'],
        registry['overlap_classes']
    )
    print(f"[OK] Risk level: {risk_level}")
    print(f"[OK] Case flags: {case_flags}")
    
    response = build_unified_response(
        densenet_results,
        yolo_results,
        risk_level,
        case_flags,
        registry['prob_thresholds'],
        registry['overlap_classes'],
        original_img_pil=original_pil
    )
    
    assert response['status'] == 'success'
    assert 'predictions' in response
    assert 'confirmed_findings' in response
    assert 'images' in response
    assert 'original_b64' in response['images']
    
    print(f"[OK] Response status: {response['status']}")
    print(f"[OK] Confirmed findings count: {len(response['confirmed_findings'])}")
    for f in response['confirmed_findings']:
        print(f"    * {f['condition']}: prob={f['prob']}, threshold={f['threshold']}, normalized_conf={f['normalized_conf']}, risk={f['risk']}")
        
    print("\n" + "=" * 60)
    print("ALL BACKEND INFERENCE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    test_backend_inference()
