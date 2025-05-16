import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import socket

from config import get_args
from dataset import FeatureDataset
from model import FusionModel
from utils import print_args

def save_checkpoint(state, epoch, is_best, save_path, model_name):
    """Save model checkpoint"""
    best_model_path = os.path.join(save_path, f'{model_name}.pt')
    
    if is_best:
        dir_path = os.path.dirname(best_model_path)
        os.makedirs(dir_path, exist_ok=True)

        torch.save(state, best_model_path)

def run_epoch(dataloader, model, tau, penalty_coefficient, optimizer=None, is_training=True, use_tqdm=False, 
              intermediate_logging=False, log_interval=500):
    """Runs a single epoch for training or validation."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_samples = 0
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    with torch.set_grad_enabled(is_training):
        loader = tqdm(dataloader) if use_tqdm else dataloader
        for idx, batch in enumerate(loader):
            visual_frame, audio_window, video_name, video_frames = batch
            current_batch_size = visual_frame.size()[0]

            visual_frame = visual_frame.to(model.device)
            audio_window = audio_window.to(model.device)

            # Repeat video frame to match audio frames (2*tau+1 times)
            visual_central_frame = visual_frame.unsqueeze(1).repeat(1, 2 * tau + 1, 1)

            outputs = model(visual_central_frame, audio_window)
            outputs = outputs.squeeze()
        
            synchronization_scores = logsoftmax(outputs)[:, tau]
            loss = -torch.sum(synchronization_scores)

            total_loss += loss.item()
            total_samples += current_batch_size

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if intermediate_logging and idx % log_interval == 0 and idx > 0:
                    avg_loss = total_loss / total_samples
                    print(f"Step [{idx}/{len(loader)}] \t Loss: {avg_loss:.6f} \t Output mean score this batch: {torch.mean(outputs).item():.3f} \t sync_score avg: {torch.mean(synchronization_scores).item():.3f}")
            
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss

def main():
    num_workers = 32
    args = get_args()
    print_args(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = FeatureDataset(
        os.path.join(args.metadata_root_path, "train_metadata.csv"), 
        os.path.join(args.data_root_path, "train"),
        tau=args.tau,
    )
    val_dataset = FeatureDataset(
        os.path.join(args.metadata_root_path, "val_metadata.csv"), 
        os.path.join(args.data_root_path, "train"),
        tau=args.tau,
    )
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    model = FusionModel().to(device)
    model.device = device
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=args.scheduler_patience
    )

    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    previous_epoch_lr = args.learning_rate

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = run_epoch(
            train_loader, model, args.tau, args.penalty_coefficient, optimizer,
            is_training=True,
            use_tqdm=args.use_tqdm,
            intermediate_logging=not args.no_intermediate_logging,
            log_interval=args.log_interval
        )
        print(f"Training - Loss: {train_loss:.6f}")

        # Validation
        val_loss = run_epoch(
            val_loader, model, args.tau, args.penalty_coefficient,
            is_training=False,
            use_tqdm=args.use_tqdm
        )
        print(f"Validation - Loss: {val_loss:.6f}")

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[-1]
        if current_lr != previous_epoch_lr:
            print(f"Learning rate has changed to {current_lr}")
            previous_epoch_lr = current_lr

        # Save checkpoint
        is_best = val_loss < best_val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'args': args
        }
        
        model_name = args.name
        save_checkpoint(checkpoint, epoch, is_best, args.save_path, model_name)

        if is_best:
            best_val_loss = val_loss
            print(f"New best model saved with validation loss: {val_loss:.6f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

    print("Training finished")

if __name__ == "__main__":
    main()