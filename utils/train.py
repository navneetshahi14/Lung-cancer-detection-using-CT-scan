import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from utils.metrics import plot_roc_auc
from copy import deepcopy
from utils.balancing import compute_class_weights
import os

def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()

    running_loss = 0
    preds, labels_list = [],[]

    for images,labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds.extend(torch.argmax(outputs,1).cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_list,preds)
    return running_loss/ len(loader) ,acc

def evaluate(model,loader,criterion,device):
    model.eval()

    running_loss = 0
    preds, labels_list = [],[]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)

            running_loss += loss.item()

            preds.extend(torch.argmax(outputs,1).cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_list, preds)
    return running_loss/ len(loader), acc, preds, labels_list


def train_model(model,train_loader,val_loader , device, epochs=10,lr=1e-4,weight_decay=1e-4,patience=5,save_dir="results/checkpoints",model_name="model"):
    # criterion = nn.CrossEntropyLoss()
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    os.makedirs(save_dir,exist_ok=True)
    
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    stopped_early = False

    best_path = os.path.join(save_dir, f"{model_name}_best.pth")

    history = {
        "train_loss":[],
        "val_loss":[],
        "train_acc":[],
        "val_acc":[]
    }

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, _ ,_ = evaluate(
            model,val_loader,criterion,device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)


        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            wait = 0

            torch.save({
                "epoch":best_epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": best_val_loss,
            },best_path)

            print(f"Best model saved -> Epoch {best_epoch}")
        
        else:
            wait += 1

        if wait >= patience:
            print(f"\n Early stopping triggered at epoch {epoch+1}")
            stopped_early = True
            break
    
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_epochs_ran": len(history["val_acc"]),
        "stopped_early": stopped_early,
    }
            
    
    print(f"\n Best Epoch: {best_epoch} | Val Acc: {best_val_acc:.4f}")

    return model, history, summary



def test_model(
    model,
    test_loader,
    device,
    class_names,
    return_details=False
):
    """
    Research-grade test function.

    Returns:
    - accuracy
    - classification report
    - confusion matrix
    - labels, preds, probs, images (optional for ROC-AUC, GradCAM, grayscale)
    """

    criterion = nn.CrossEntropyLoss()

    model.eval()

    running_correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []
    sample_images = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # store few images for GradCAM / grayscale
            if len(sample_images) < 10:
                sample_images.extend(images.cpu())

    test_acc = running_correct / total

    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    if return_details:
        return (
            test_acc,
            report,
            cm,
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            sample_images,
        )

    return test_acc, report, cm
