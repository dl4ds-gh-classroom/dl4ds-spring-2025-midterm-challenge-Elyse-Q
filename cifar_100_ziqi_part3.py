import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total

    return train_loss, train_acc
################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): 
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    CONFIG = {
        "model": "MyModel",   # Change name when using a different model
        "batch_size": 512, # run batch size finder to find optimal batch size
        "learning_rate": 0.001,
        "epochs": 50,  # Train for longer in a real scenario
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "-sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # random crop, data augmentation
        transforms.RandomHorizontalFlip(),          # random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size

    trainset, valset = random_split(trainset, [train_size, val_size])

    ### TODO -- define loaders and test set
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2) 

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    lr_candidates = [0.001, 0.0005, 0.0001]
    wd_candidates = [1e-5, 1e-4, 1e-3]
    dropout_candidates = [0.1, 0.2, 0.3]
    tuning_epochs = 5
    best_hp_val_acc = 0.0
    best_hp_config = {"lr": None, "weight_decay": None, "dropout": None}
    for lr in lr_candidates:
        for wd in wd_candidates:
            for dropout in dropout_candidates:
                print(f"\nTesting lr: {lr}, weight_decay: {wd}, dropout: {dropout}")
                model_hp = torchvision.models.resnet50(pretrained=True)
                in_features = model_hp.fc.in_features
                # create new fc layer according to current dropout 
                model_hp.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, 100)
                )
                model_hp = model_hp.to(CONFIG["device"])
                criterion_hp = nn.CrossEntropyLoss()
                optimizer_hp = optim.Adam(model_hp.parameters(), lr=lr, weight_decay=wd)
                for epoch in range(tuning_epochs):
                    train(epoch, model_hp, trainloader, optimizer_hp, criterion_hp, CONFIG)
                    _, val_acc = validate(model_hp, valloader, criterion_hp, CONFIG["device"])
                print(f"Combination (lr={lr}, wd={wd}, dropout={dropout}) -> val_acc: {val_acc:.2f}%")
                if val_acc > best_hp_val_acc:
                    best_hp_val_acc = val_acc
                    best_hp_config["lr"] = lr
                    best_hp_config["weight_decay"] = wd
                    best_hp_config["dropout"] = dropout

    print(f"\nBest Hyperparameters: {best_hp_config}, Best val_acc: {best_hp_val_acc:.2f}%")

    model = torchvision.models.resnet50(pretrained=True)
    # Add dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=best_hp_config["dropout"]),
        nn.Linear(in_features, 100)
    )
    # model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(CONFIG["device"])

    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    
    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_hp_config["lr"], weight_decay=best_hp_config["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=17, gamma=0.5)
    
    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients
    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0
    # Early Stopping parameters
    early_stopping_patience = 7
    no_improve_count = 0
    best_val_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()
        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print("Early stopping triggered!")
                break
        # Save the best model (based on validation accuracy)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), "best_model.pth")
        #     wandb.save("best_model.pth") # Save to wandb as well

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()
