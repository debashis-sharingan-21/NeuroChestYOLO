import sys
import os
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app

def test_api():
    print("Testing FastAPI app with TestClient...")
    with TestClient(app) as client:
        # Test root endpoint
        res = client.get("/")
        assert res.status_code == 200, f"Root endpoint failed: {res.text}"
        print(f"[OK] Root endpoint response: {res.json()}")

        # Test health endpoint
        res = client.get("/health")
        assert res.status_code == 200, f"Health endpoint failed: {res.text}"
        data = res.json()
        assert data["models_loaded"] is True, "Models should be loaded"
        assert len(data["available_classes"]) == 15, "Expected 15 classes"
        print(f"[OK] Health endpoint: status={data['status']}, models_loaded={data['models_loaded']}, classes_count={len(data['available_classes'])}")

        # Test predict endpoint with sample image
        sample_img_path = os.path.join(BASE_DIR, 'frontend', 'public', 'testing_images', '36204.png')
        with open(sample_img_path, 'rb') as f:
            files = {"file": ("36204.png", f, "image/png")}
            res = client.post("/predict", files=files)
            
        assert res.status_code == 200, f"Predict endpoint failed: {res.text}"
        pred = res.json()
        assert pred["status"] == "success", "Expected status == success"
        assert "predictions" in pred
        assert "confirmed_findings" in pred
        assert "images" in pred
        assert "risk_level" in pred
        print(f"[OK] Predict endpoint returned 200 OK.")
        print(f"[OK] Risk level: {pred['risk_level']}")
        print(f"[OK] Confirmed findings: {[c['condition'] for c in pred['confirmed_findings']]}")
        print(f"[OK] Base64 images generated: {list(pred['images'].keys())}")

    print("\nAPI ENDPOINT VERIFICATION COMPLETED WITH 100% SUCCESS!")

if __name__ == '__main__':
    test_api()
