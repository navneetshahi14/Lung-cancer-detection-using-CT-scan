import torch


def load_trained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"📊 Saved Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")

    return model
