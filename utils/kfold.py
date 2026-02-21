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
    batch_size=8,
):

    torch.manual_seed(42)
    np.random.seed(42)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = np.array(dataset.targets)

    fold_accuracies, fold_f1 = [], []

    print(f"\n===== {k}-Fold Cross Validation Started =====")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        print(f"\n🔁 Fold {fold+1}/{k}")

        # -------- Subsets --------
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset   = torch.utils.data.Subset(dataset, val_idx)

        # -------- Loaders --------
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )

        # -------- Fresh model --------
        model = model_builder().to(device)

        # -------- Train --------
        model, history, summary = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs,
            model_name=f"kfold_fold{fold+1}",
        )

        # -------- Evaluate --------
        acc, report, cm, labels_out, preds, probs, _ = test_model(
            model,
            val_loader,
            device,
            class_names,
            return_details=True,
        )

        _, _, f1 = compute_full_metrics(labels_out, preds)[1:]

        fold_accuracies.append(acc)
        fold_f1.append(f1)

        print(f"Fold {fold+1} → Acc: {acc:.4f} | F1: {f1:.4f}")

    # -------- Final stats --------
    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    mean_f1, std_f1   = np.mean(fold_f1), np.std(fold_f1)

    print("\n===== FINAL K-Fold RESULT =====")
    print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1-Score : {mean_f1:.4f} ± {std_f1:.4f}")

    return {
        "fold_acc": fold_accuracies,
        "fold_f1": fold_f1,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }
