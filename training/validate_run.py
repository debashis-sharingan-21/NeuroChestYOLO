"""
Minimal Training Loss Convergence & Checkpoint Verification Script.
Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
from training.model import DenseNetCustom
from training.dataset import NIH_15_CLASSES

def verify_training_convergence():
    print("=" * 65)
    print("Neuro CXR — Training Engine Verification & Dry-Run")
    print("Maintainer: Debashis Mohapatra (2026)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Hardware Device: {device}")

    # 1. Instantiate model
    num_classes = len(NIH_15_CLASSES)
    model = DenseNetCustom(num_classes=num_classes, pretrained=False).to(device)
    print(f"[2/4] Model Architecture: DenseNetCustom with Dual Pooling Head ({num_classes} classes)")

    # 2. Synthetic Batch for Overfitting Convergence Check
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    # Synthetic ground truth: multi-label binary indicators
    dummy_targets = torch.zeros(batch_size, num_classes, device=device)
    dummy_targets[0, 1] = 1.0  # Cardiomegaly
    dummy_targets[0, 2] = 1.0  # Effusion
    dummy_targets[1, 6] = 1.0  # Pneumonia
    dummy_targets[2, 7] = 1.0  # Pneumothorax
    dummy_targets[3, 14] = 1.0 # No Finding

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("[3/4] Running 10 iterative gradient steps to verify convergence...")
    initial_loss = None
    final_loss = None

    model.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        logits = model(dummy_input)
        loss = criterion(logits, dummy_targets)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if step == 1:
            initial_loss = loss_val
        final_loss = loss_val
        print(f"      Step {step:02d}/10 — Loss: {loss_val:.5f}")

    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {final_loss}"
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"      Convergence Confirmed: Loss dropped by {loss_reduction:.2f}% (from {initial_loss:.4f} to {final_loss:.4f})")

    # 3. Checkpoint Saving and Reloading Verification
    print("[4/4] Verifying Checkpoint Serialization & Weight Restoration...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pth")
        payload = {
            "epoch": 1,
            "label_columns": NIH_15_CLASSES,
            "model_state_dict": model.state_dict(),
            "val_loss": final_loss,
            "author": "Debashis Mohapatra",
            "year": 2026
        }
        torch.save(payload, ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file was not written to disk!"

        # Reload
        reloaded = torch.load(ckpt_path, map_location=device)
        test_model = DenseNetCustom(num_classes=num_classes, pretrained=False).to(device)
        test_model.load_state_dict(reloaded["model_state_dict"])
        model.eval()
        test_model.eval()

        with torch.no_grad():
            out_orig = model(dummy_input)
            out_reloaded = test_model(dummy_input)
            diff = (out_orig - out_reloaded).abs().max().item()
            assert diff < 1e-6, f"Checkpoint restoration parity failed with diff: {diff}"

    print("      Checkpoint Serialization & State Dict parity verified (max diff < 1e-6).")
    print("-" * 65)
    print("ALL SANITY CHECKS PASSED: Model, Loss, Optimizer & Checkpointing Validated.")
    print("=" * 65)

if __name__ == "__main__":
    verify_training_convergence()
