import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import optuna
import os
from model import RNAHybridModel, MaskedSNRLoss
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from tqdm import tqdm


# ---------------------------- Dataset ----------------------------

class RNADataset(Dataset):
    def __init__(self, seq_file, label_file, msa_features=None):
        self.seq_df = pd.read_csv(seq_file)
        self.label_df = pd.read_csv(label_file)
        self.ids = self.seq_df["target_id"].unique()
        self.msa_features = msa_features

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        target_id = self.ids[idx]
        x = self.seq_df[self.seq_df["target_id"] == target_id].drop(columns=["target_id"]).values.astype(np.float32)
        y = self.label_df[self.label_df["target_id"] == target_id].drop(columns=["target_id"]).values.astype(np.float32)
        if self.msa_features:
            msa = self.msa_features[target_id]
            x = np.concatenate([x, msa], axis=-1)
        return torch.tensor(x), torch.tensor(y)


# ---------------------------- Train/Val Utils ----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x, None, None, x.shape[1])
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x, None, None, x.shape[1])
            loss = criterion(preds, y)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return total_loss / len(loader), np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------- Optuna Objective ----------------------------

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = RNAHybridModel(input_dim=10, hidden_dim=hidden_dim, output_per_residue=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.MSELoss()

    train_dataset = RNADataset("train_sequences.csv", "train_labels.csv")
    val_dataset = RNADataset("validation_sequences.csv", "validation_labels.csv")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(30):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


# ---------------------------- Main ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print("Best Params:", best_params)

    model = RNAHybridModel(input_dim=10, hidden_dim=best_params["hidden_dim"], output_per_residue=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.MSELoss()

    train_dataset = RNADataset("train_sequences.csv", "train_labels.csv")
    val_dataset = RNADataset("validation_sequences.csv", "validation_labels.csv")
    test_dataset = RNADataset("test_sequences.csv", "test_labels.csv")

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"])

    best_model = None
    best_val_loss = float("inf")
    patience = 8
    counter = 0

    snapshots = []

    for epoch in range(50):
        print(f"Epoch {epoch+1}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            counter = 0
            snapshots.append(deepcopy(model.state_dict()))
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "best_model.pth")

    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    print(f"Test MSE: {test_loss:.4f}")


