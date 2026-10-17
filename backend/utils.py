import io
import base64
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Preprocessing for DenseNet as per Section 15
DENSENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes, max_dim=256):
    """Decode image bytes, downsample to max_dim (default 256), and prepare for DenseNet."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if max(img.size) > max_dim:
        resample_filter = getattr(Image.Resampling, 'BILINEAR', Image.BILINEAR)
        img.thumbnail((max_dim, max_dim), resample_filter)
    tensor = DENSENET_TRANSFORM(img).unsqueeze(0)  # [1,3,224,224]
    return tensor, img

def numpy_to_b64(img_rgb, quality=75):
    """Convert numpy RGB image to base64 string for JSON response."""
    # Convert RGB (PIL/Standard) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

def pil_to_numpy(pil_img):
    """Convert PIL image to numpy RGB array."""
    return np.array(pil_img)

def normalize_label(name):
    """Normalize label name (underscore to space) for cross-model matching."""
    return name.replace('_', ' ')
