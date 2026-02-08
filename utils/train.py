import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def train_model(model,train_loader,val_loader, device, epochs=10,lr=1e-4,weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

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
    
    return model, history



def test_model(model, test_loader, device, class_names):

    criterion = nn.CrossEntropyLoss()

    _,test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device
    )

    report = classification_report(labels, preds, target_names=class_names)
    cm = confusion_matrix(labels, preds)

    return test_acc, report, cm

