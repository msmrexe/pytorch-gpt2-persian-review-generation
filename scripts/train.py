"""
Main training script for the sentiment-controlled GPT-2 model.

This script handles:
1. Setting up logging.
2. Parsing command-line arguments.
3. Loading tokenizer and data.
4. Initializing the model, optimizer, and scheduler.
5. Running the training and evaluation loop.
6. Saving the best model and plotting results.
"""

import argparse
import logging
import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import setup_logging, download_data, load_and_preprocess_data, plot_losses
from src.dataset import get_tokenizer, create_dataloaders
from src.config import GPT2Config
from src.model import GPT2

# Setup logger
logger = setup_logging()

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a sentiment-controlled GPT-2 model from scratch."
    )
    
    # Data and Tokenizer
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default="mohammad1ziyar/cleaned-snappfood-persian-sentiment-analysis",
        help="Kaggle dataset name (e.g., user/dataset)."
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default="data/",
        help="Directory to download and store data."
    )
    parser.add_argument(
        '--tokenizer_name', 
        type=str, 
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Hugging Face tokenizer model name."
    )
    
    # Model Config
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=128,
        help="Max sequence length (n_positions)."
    )
    parser.add_argument(
        '--n_embd', 
        type=int, 
        default=192,
        help="Embedding dimension."
    )
    parser.add_argument(
        '--n_layer', 
        type=int, 
        default=3,
        help="Number of transformer blocks."
    )
    parser.add_argument(
        '--n_head', 
        type=int, 
        default=3,
        help="Number of attention heads."
    )
    
    # Training
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=5,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help="Training and validation batch size."
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.01,
        help="AdamW weight decay."
    )
    parser.add_argument(
        '--no_scheduler', 
        action='store_true',
        help="Disable the cosine annealing scheduler."
    )
    parser.add_argument(
        '--clip_grad', 
        type=float, 
        default=1.0,
        help="Max norm for gradient clipping."
    )
    parser.add_argument(
        '--log_interval', 
        type=int, 
        default=100,
        help="Log training progress every N steps."
    )
    
    # System
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu'). Autodetects if None."
    )
    parser.add_argument(
        '--model_save_path', 
        type=str, 
        default="models/best_gpt2_model.pt",
        help="Path to save the best model checkpoint."
    )
    
    return parser.parse_args()

def train_epoch(
    model: nn.Module, 
    data_loader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    device: torch.device, 
    log_interval: int,
    clip_grad: float
) -> tuple:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    step_nums = []
    step_losses = []
    
    pbar = tqdm(data_loader, desc=f"Training", unit="batch")
    
    for step, batch in enumerate(pbar):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs['loss']
            
            if loss is None:
                logger.warning(f"Step {step}: Loss is None. Skipping batch.")
                continue
                
            loss.backward()
            
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
            total_loss += loss.item()
            current_avg_loss = total_loss / (step + 1)
            
            pbar.set_postfix(
                loss=f'{loss.item():.4f}', 
                avg_loss=f'{current_avg_loss:.4f}',
                lr=f'{scheduler.get_last_lr()[0]:.2e}' if scheduler else f'{optimizer.param_groups[0]["lr"]:.2e}'
            )
            
            if step % log_interval == 0 and step > 0:
                step_nums.append(step)
                step_losses.append(loss.item())
        
        except Exception as e:
            logger.error(f"Error during training step {step}: {e}")
            logger.error(f"Input shapes: {input_ids.shape if 'input_ids' in locals() else 'N/A'}")
            continue # Skip batch

    avg_epoch_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    return avg_epoch_loss, step_nums, step_losses

def evaluate(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device, 
    log_interval: int
) -> tuple:
    """Runs a single evaluation epoch."""
    model.eval()
    total_loss = 0.0
    step_nums = []
    step_losses = []
    
    pbar = tqdm(data_loader, desc=f"Evaluating", unit="batch")
    
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs['loss']
                
                if loss is None:
                    continue
                    
                total_loss += loss.item()
                current_avg_loss = total_loss / (step + 1)
                
                pbar.set_postfix(
                    loss=f'{loss.item():.4f}', 
                    avg_loss=f'{current_avg_loss:.4f}'
                )
                
                if step % log_interval == 0:
                    step_nums.append(step)
                    step_losses.append(loss.item())

            except Exception as e:
                logger.error(f"Error during evaluation step {step}: {e}")
                continue # Skip batch
                
    avg_epoch_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    return avg_epoch_loss, step_nums, step_losses

def main():
    """Main training function."""
    args = get_args()
    
    logger.info("Starting training script...")
    logger.info(f"Arguments: {vars(args)}")
    
    # --- 1. Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- 2. Load Data ---
    try:
        data_file = download_data(args.dataset_name, args.data_dir)
        raw_corpus = load_and_preprocess_data(data_file)
    except Exception as e:
        logger.fatal(f"Failed to load data: {e}. Exiting.")
        return

    # --- 3. Load Tokenizer ---
    try:
        tokenizer = get_tokenizer(args.tokenizer_name)
    except Exception as e:
        logger.fatal(f"Failed to load tokenizer: {e}. Exiting.")
        return

    # --- 4. Create DataLoaders ---
    train_loader, val_loader = create_dataloaders(
        raw_corpus,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    # --- 5. Initialize Model ---
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.max_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2(config)
    model.to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- 6. Setup Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scheduler = None
    if not args.no_scheduler:
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps
        )
        logger.info(f"Using CosineAnnealingLR scheduler with {total_steps} total steps.")
    else:
        logger.info("Scheduler is disabled.")

    # --- 7. Training Loop ---
    epoch_train_losses = []
    epoch_val_losses = []
    all_train_step_losses = []
    all_train_step_nums = []
    all_val_step_losses = []
    all_val_step_nums = []
    best_val_loss = float('inf')
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Batches per epoch: {len(train_loader)} (train), {len(val_loader)} (val)")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            
            train_loss, train_steps, train_losses = train_epoch(
                model, train_loader, optimizer, scheduler, device, 
                args.log_interval, args.clip_grad
            )
            epoch_train_losses.append(train_loss)
            
            offset = epoch * len(train_loader)
            all_train_step_nums.extend([s + offset for s in train_steps])
            all_train_step_losses.extend(train_losses)
            
            logger.info(f"Epoch {epoch + 1} Training complete.")
            
            val_loss, val_steps, val_losses = evaluate(
                model, val_loader, device, args.log_interval
            )
            epoch_val_losses.append(val_loss)
            
            val_offset = epoch * len(val_loader)
            all_val_step_nums.extend([s + val_offset for s in val_steps])
            all_val_step_losses.extend(val_losses)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"Time: {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'config': config, # Save the config
                    'tokenizer_name': args.tokenizer_name # Save tokenizer name
                }, args.model_save_path)
                logger.info(f"Saved best model to {args.model_save_path}")
            
            logger.info("-" * 40)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving plots and exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
    finally:
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training finished in {total_time / 60:.2f} minutes.")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # --- 8. Plot and Save Results ---
        if epoch_train_losses:
            plot_losses(
                epochs=len(epoch_train_losses),
                epoch_train_losses=epoch_train_losses,
                epoch_val_losses=epoch_val_losses,
                all_train_step_nums=all_train_step_nums,
                all_train_step_losses=all_train_step_losses,
                all_val_step_nums=all_val_step_nums,
                all_val_step_losses=all_val_step_losses,
                save_path=os.path.join(
                    os.path.dirname(args.model_save_path), 
                    "training_loss_plots.png"
                )
            )
        else:
            logger.warning("No epochs completed. Skipping plot generation.")

if __name__ == "__main__":
    main()
