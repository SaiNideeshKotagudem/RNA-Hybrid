import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from _model import RNAHybridModel, MaskedSNRLoss
import optuna
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR


# -------- Dataset --------
class RNASeqDataset(Dataset):
    def __init__(self, seq_csv, label_csv, msa_cols=None, phase='train'):
        """
        seq_csv: path to sequences csv
        label_csv: path to labels csv
        msa_cols: list of column names for MSA features if present
        phase: 'train', 'val', 'test'
        """
        self.seq_df = pd.read_csv(seq_csv)
        self.label_df = pd.read_csv(label_csv)
        self.phase = phase
        
        # encode resname if present in labels (convert string to int)
        if 'resname' in self.label_df.columns:
            self.resname_encoder = LabelEncoder()
            self.label_df['resname'] = self.resname_encoder.fit_transform(self.label_df['resname'])

        # Extract input features
        # Assume base 10 input features + optional MSA columns appended
        self.base_features = [col for col in self.seq_df.columns if col not in ('resname', 'resid')]
        if msa_cols:
            self.base_features += msa_cols
        self.inputs = self.seq_df[self.base_features].values.astype(np.float32)

        # Targets columns differ by phase:
        if phase == 'train':
            # Predict 1 residue x (x,y,z + resname + resid) = 5 columns
            target_cols = ['x1', 'y1', 'z1', 'resname', 'resid']
        elif phase == 'val':
            # Predict 40 residues x 3 + 2
            coords = [f'{ax}{i}' for i in range(1,41) for ax in ('x','y','z')]
            target_cols = coords + ['resname', 'resid']
        else:
            # test phase: 5 residues x 3 + 2
            coords = [f'{ax}{i}' for i in range(1,6) for ax in ('x','y','z')]
            target_cols = coords + ['resname', 'resid']

        # Make sure target columns exist
        for col in target_cols:
            if col not in self.label_df.columns:
                raise ValueError(f"Missing target column {col} in {phase} labels csv")

        self.targets = self.label_df[target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx])
        y = torch.tensor(self.targets[idx])
        return x, y


# -------- Utilities --------
def collate_batch(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys


def load_data_loaders(batch_size, msa_cols=None):
    train_dataset = RNASeqDataset('train_sequences.csv', 'train_labels.csv', msa_cols=msa_cols, phase='train')
    val_dataset = RNASeqDataset('validation_sequences.csv', 'validation_labels.csv', msa_cols=msa_cols, phase='val')
    test_dataset = RNASeqDataset('test_sequences.csv', 'test_labels.csv', msa_cols=msa_cols, phase='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, val_loader, test_loader


# -------- Training + Validation Loops --------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, bpp_edge_index=None, bpp_edge_attr=None, bpp_num_nodes=inputs.shape[1])  # TODO: Adjust if using BPP edges

        # MSE Loss on coordinates only (first 3 columns usually)
        coord_pred = outputs[:, :, :3]  # assuming coords are first 3 outputs per residue
        coord_target = targets[:, :3]

        loss = criterion(coord_pred, coord_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, bpp_edge_index=None, bpp_edge_attr=None, bpp_num_nodes=inputs.shape[1])

            coord_pred = outputs[:, :, :3]
            coord_target = targets[:, :3]

            loss = criterion(coord_pred, coord_target)
            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


# -------- Early Stopping --------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# -------- Snapshot Ensemble Checkpointing --------
class SnapshotEnsemble:
    def __init__(self, max_snapshots=5, save_dir='snapshots'):
        self.max_snapshots = max_snapshots
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.snapshots = []

    def save(self, model, optimizer, epoch, val_loss):
        checkpoint_path = os.path.join(self.save_dir, f'snapshot_epoch{epoch}_loss{val_loss:.4f}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)

        self.snapshots.append((val_loss, checkpoint_path))
        self.snapshots.sort(key=lambda x: x[0])
        if len(self.snapshots) > self.max_snapshots:
            _, to_remove = self.snapshots.pop(-1)
            os.remove(to_remove)


# -------- Objective function for Optuna --------
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data with batch size
    train_loader, val_loader, _ = load_data_loaders(batch_size)

    # Initialize model with hidden dim
    model = RNAHybridModel(input_dim=10, hidden_dim=hidden_dim, output_per_residue=5)  # training output shape
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    early_stopping = EarlyStopping(patience=5, verbose=True)
    snapshot_ensemble = SnapshotEnsemble(max_snapshots=3)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        snapshot_ensemble.save(model, optimizer, epoch, val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == "__main__":
    # Run Optuna hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print(f"Best trial: {study.best_trial.params}")
