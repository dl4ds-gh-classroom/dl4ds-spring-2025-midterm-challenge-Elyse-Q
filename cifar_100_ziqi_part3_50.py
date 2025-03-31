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
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
    
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

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here

        # parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calculate the loss
        loss = criterion(outputs, labels)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # update the loss 
        running_loss += loss.item()

        # calculate the accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total

    print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # update the loss
            running_loss += loss.item()

            # calculate the accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.

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
        transforms.RandomCrop(32, padding=4),       # 随机裁剪（数据增强）
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
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
    # lrs = [0.001, 0.0005, 0.0001]
    # wds = [1e-5, 1e-4, 1e-3]
    # tuning_epochs = 5  # use small epoch to tune the model
    # best_hp_val_acc = 0.0
    # best_hp_config = {"lr": None, "weight_decay": None}

    # for lr in lrs:
    #     for wd in wds:
    #         print(f"\nTesting learning rate: {lr}, weight_decay: {wd}")
    #         model_hp = torchvision.models.resnet50(pretrained=True)
    #         # Add dropout in the final layer for regularization
    #         in_features = model_hp.fc.in_features
    #         model_hp.fc = nn.Sequential(
    #             nn.Dropout(p=0.1),
    #             nn.Linear(in_features, 100)
    #         )
    #         # model_hp.fc = nn.Linear(model_hp.fc.in_features, 100)
    #         model_hp = model_hp.to(CONFIG["device"])
    #         criterion_hp = nn.CrossEntropyLoss()
    #         optimizer_hp = optim.Adam(model_hp.parameters(), lr=lr, weight_decay=wd)
    #         for epoch in range(tuning_epochs):
    #             _ , _ = train(epoch, model_hp, trainloader, optimizer_hp, criterion_hp, CONFIG)
    #             _, val_acc = validate(model_hp, valloader, criterion_hp, CONFIG["device"])
    #         print(f"Combination (lr={lr}, weight_decay={wd}) val_acc: {val_acc:.2f}%")
    #         if val_acc > best_hp_val_acc:
    #             best_hp_val_acc = val_acc
    #             best_hp_config["lr"] = lr
    #             best_hp_config["weight_decay"] = wd

    # print(f"\nbest hp: {best_hp_config}, val_acc: {best_hp_val_acc:.2f}%")
    lr_candidates = [0.001, 0.0005, 0.0001]
    wd_candidates = [1e-5, 1e-4, 1e-3]
    dropout_candidates = [0.1, 0.2, 0.3]
    tuning_epochs = 7 
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
    # print("\nModel summary (Pretrained ResNet50):")
    # print(model, "\n")
    # print("\nModel summary (Pretrained ResNet18):")
    # print(model, "\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
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
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=17, gamma=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    
    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0
    # Early Stopping 参数
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