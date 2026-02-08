import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_history(history, save_path =None):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path,dpi=300)

    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def save_results_csv(
    results_dict,
    save_path,
    preprocess_cfg=None,
    epochs=None
):
    rows = []

    for model_name, acc in results_dict.items():

        row = {
            "Model": model_name,
            "Accuracy": acc,
        }

        if epochs is not None:
            row["Epochs"] = epochs

        if preprocess_cfg is not None:
            for k, v in preprocess_cfg.items():
                row[f"prep_{k}"] = v

        rows.append(row)

    df_new = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(save_path, index=False)

    print(f"✅ Results saved → {save_path}")
    return df



def plot_model_comparison(results_dict, save_path=None):

    models = list(results_dict.keys())
    accs = list(results_dict.values())

    plt.figure(figsize=(8, 5))

    sns.barplot(x=models, y=accs)

    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Model Comparison")

    plt.xticks(rotation=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()