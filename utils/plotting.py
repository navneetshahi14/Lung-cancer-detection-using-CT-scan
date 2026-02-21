import os 
import matplotlib.pyplot as plt

def plot_training_curve(history,save_path,title="Training Curve"):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    epochs = range(1,len(history["train_loss"])+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📈 Training curves saved → {save_path}")