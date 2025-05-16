import torch
from train import RNAHybridModel, load_data_loaders, train_epoch, validate_epoch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

def train_final_model(best_params, num_epochs=50, patience=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = best_params['batch_size']
    hidden_dim = best_params['hidden_dim']
    lr = best_params['lr']

    train_loader, val_loader, _ = load_data_loaders(batch_size=batch_size)

    model = RNAHybridModel(input_dim=10, hidden_dim=hidden_dim, output_per_residue=5)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")



if __name__ == "__main__":
    import optuna
    from train import objective

    # Step 1: Run Optuna search
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print(f"Best hyperparameters found: {study.best_trial.params}")

    # Step 2: Train final model with best hyperparameters
    train_final_model(study.best_trial.params)
