import torch
import torch.nn as nn
import json
import types
import os
import numpy
import torch.nn.functional as F
from torchvision import models
from ultralytics import YOLO

# Allowlist numpy scalars for torch.load security (required for PyTorch 2.6+)
try:
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
except:
    pass

# Custom Architecture from nih-nb.ipynb
class DenseNetCustom(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetCustom, self).__init__()
        # Initialize with standard densenet but extract features
        dn = models.densenet121(weights=None)
        self.backbone = dn.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # Linear layer is 1024 * 2 due to dual pooling (avg + max)
        self.classifier = nn.Linear(1024 * 2, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = F.relu(features, inplace=False) # Critical for GradCAM
        ap = self.avgpool(out).flatten(1)
        mp = self.maxpool(out).flatten(1)
        combined = torch.cat([ap, mp], dim=1)
        return self.classifier(combined)

def load_all_models(
    nih_pth_path       = 'models/nih-model.pth',
    thresholds_path    = 'models/thresholds_and_metrics.json',
    yolo_pt_path       = 'models/best_chestxdet.pt',
    yolo_config_path   = 'models/yolo_pipeline_config.json',
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nih_pth_path = os.path.join(base_dir, nih_pth_path)
    thresholds_path = os.path.join(base_dir, thresholds_path)
    yolo_pt_path = os.path.join(base_dir, yolo_pt_path)
    yolo_config_path = os.path.join(base_dir, yolo_config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load DenseNet Custom Model
    if not os.path.exists(nih_pth_path):
        raise FileNotFoundError(f"Model file not found: {nih_pth_path}")
        
    checkpoint = torch.load(nih_pth_path, map_location=device, weights_only=False)
    label_cols = checkpoint['label_columns']

    # Initialize the custom architecture
    densenet = DenseNetCustom(len(label_cols))
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        densenet.load_state_dict(checkpoint['model_state_dict'])
    else:
        densenet.load_state_dict(checkpoint)

    densenet.to(device)
    densenet.eval()

    # 2. Load Thresholds
    thresh_data = json.load(open(thresholds_path))
    prob_thresholds = thresh_data['probability_thresholds']
    cam_thresholds = thresh_data['cam_thresholds']

    # 3. Load YOLOv8 Detection Model
    yolo = YOLO(yolo_pt_path)
    yolo_config = json.load(open(yolo_config_path))

    return {
        'densenet': densenet,
        'device': device,
        'label_columns': label_cols,
        'prob_thresholds': prob_thresholds,
        'cam_thresholds': cam_thresholds,
        'yolo': yolo,
        'id_to_class': yolo_config['id_to_class'],
        'overlap_classes': set(yolo_config['overlap_classes']),
        'yolo_conf': yolo_config['conf_thresh'],
        'yolo_iou': yolo_config['iou_thresh'],
        'yolo_imgsz': yolo_config.get('img_size_train', 800)
    }
