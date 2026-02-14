import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from utils.train import train_model, test_model
from utils.metrics import compute_full_metrics

def run_kfold_training(
    model_builder,
    dataset,
    device,
    class_names,
    k=5,
    epochs=10,
):
    """
    model_builder → function that returns new model
    dataset → full dataset (ImageFolder)
    """

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    labels = np.array(dataset.targets)

    fold_accuracies = []
    fold_f1 = []

    print(f"\n===== {k}-Fold Cross Validation Started =====")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        print(f"\n🔁 Fold {fold+1}/{k}")

        # Subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False)

        # Fresh model every fold
        model = model_builder().to(device)

        # Train
        model, history, summary = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            model_name=f"kfold_fold{fold+1}"
        )

        # Validate as test
        acc, report, cm, labels_out, preds, probs, _ = test_model(
            model,
            val_loader,
            device,
            class_names,
            return_details=True
        )

        precision, recall, f1 = compute_full_metrics(labels_out, preds)[1:]

        fold_accuracies.append(acc)
        fold_f1.append(f1)

        print(f"Fold {fold+1} Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Final stats
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    mean_f1 = np.mean(fold_f1)
    std_f1 = np.std(fold_f1)

    print("\n ===== K-Fold Result =====")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

    return {
        "fold_acc": fold_accuracies,
        "fold_f1": fold_f1,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }

