import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def compute_full_metrics(labels,preds):
    accuracy = np.mean(np.array(labels) == np.array(preds))
    precision = precision_score(labels,preds, average="weighted",zero_division=0)
    recall = recall_score(labels,preds,average="weighted",zero_division=0)
    f1 = f1_score(labels,preds,average="weighted",zero_division=0)

    return accuracy,precision,recall,f1

def plot_roc_auc(y_true,y_probs,class_names,save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    n_classes = len(class_names)

    plt.figure(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ ROC-AUC saved → {save_path}")



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
    epochs=None,
    precision=None,
    recall=None,
    f1=None,
    auc_score=None
):
    rows = []

    for model_name, acc in results_dict.items():

        row = {
            "Model": model_name,
            "Accuracy": acc,
        }

        if precision is not None:
            row["Precision"] = precision

        if recall is not None:
            row["Recall"] = recall

        if f1 is not None:
            row["F1"] = f1

        if auc_score is not None:
            row["ROC_AUC"] = auc_score

        if epochs is not None:
            row["Epochs"] = epochs

        if preprocess_cfg is not None:
            for k, v in preprocess_cfg.items():
                row[f"prep_{k}"] = v

        rows.append(row)

    df_new = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ---- append instead of overwrite ----
    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(save_path, index=False)

    print(f"✅ Detailed experiment results saved → {save_path}")
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


def save_grayscale_samples(images, save_dir, max_images=5):
    """
    Saves grayscale versions of sample images.
    """

    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(len(images), max_images)):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(os.path.join(save_dir, f"gray_{i}.png"), gray)

    print(f"✅ Grayscale samples saved → {save_dir}")

def save_epoch_history_excel(history, model_name, save_path="results/experiment_logs/experiment_logs.xlsx"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame({
        "Epoch":list(range(1,len(history["train_loss"]) + 1)),
        "Train Loss":history["train_loss"],
        "Val Loss":history["val_loss"],
        "Train Acc":history["val_acc"],
        "LR": history.get("lr",[None] * len(history["train_loss"]))
    })

    if os.path.exists(save_path):
        with pd.ExcelWriter(save_path,engine="openpyxl",mode="a",if_sheet_exists="replace") as writer:
            df.to_excel(writer,sheet_name=model_name,index=False)
    else:
        with pd.ExcelWriter(save_path,engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=model_name,index=False)

    print(f"Epoch history saved -> {save_path} | Sheet: {model_name}")

def save_training_summary_excel(summary, model_name, save_path="results/training_summary.xlsx"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_new = pd.DataFrame([{
        "Model": model_name,
        "Best Epoch": summary["best_epoch"],
        "Best Val Acc": summary["best_val_acc"],
        "Best Val Loss": summary["best_val_loss"],
        "Total Epochs Ran": summary["total_epochs_ran"],
        "Stopped Early": summary["stopped_early"],
    }])

    if os.path.exists(save_path):
        df_old = pd.read_excel(save_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(save_path, index=False)

    print(f"📊 Training summary saved → {save_path}")