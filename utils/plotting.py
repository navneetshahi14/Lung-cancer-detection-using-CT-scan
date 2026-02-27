import os 
import matplotlib.pyplot as plt

def plot_training_curve(history, save_path, title="Training Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 8))

    # 🔹 Train Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # 🔹 Validation Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    # 🔹 Train Accuracy
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    # 🔹 Validation Accuracy
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    # 🔥 Validation F1 (Important)
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history["val_f1"], label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📈 Training curves saved → {save_path}")