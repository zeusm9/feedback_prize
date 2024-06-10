import gc
import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from losses import MCRMSE

class Trainer:
    def __init__(self, dataloaders, model, loss_fn, scaler, optimizer, scheduler):
        self.train_loader, self.valid_loader = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self):
        self.model.train()
        train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        train_preds, train_targets = [], []
        for _, (inputs, targets) in train_pbar:

            with autocast(enabled=True):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_item = loss.item()

                train_pbar.set_description('loss: {:.2f}'.format(loss_item))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_preds.extend(outputs.cpu().detach().numpy().tolist())
        
        del outputs, targets, inputs, loss_item, loss
        gc.collect()
        return train_preds, train_targets
    
    @torch.no_grad()
    def validate_one_epoch(self):
        
        self.model.eval()
        valid_pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        valid_preds, valid_targets = [], []
        for _, (inputs, targets) in valid_pbar:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            valid_pbar.set_description(desc=f"val_loss: {loss.item():.4f}")

            valid_targets.extend(targets.cpu().detach().numpy().tolist())
            valid_preds.extend(outputs.cpu().detach().numpy().tolist())
        
        del outputs, inputs, targets, loss
        gc.collect()
        torch.cuda.empty_cache()
        return valid_preds, valid_targets
    
    def fit(self, epochs: int = 10, output_dir: str = '', custom_name: str = 'model.pth'):
        best_loss = int(1e+7)
        best_preds = None

        for ep in range(epochs):
            print(f"{'='*20} Epoch: {ep+1} / {epochs} {'='*20}")

            train_preds, train_targets = self.train_one_epoch()
            train_mcrmse = MCRMSE(train_targets, train_preds)
            print(f"Training MCRMSE: {train_mcrmse:.4f}")

            valid_preds, valid_targets = self.validate_one_epoch()
            valid_mcrmse = MCRMSE(valid_targets, valid_preds)
            print(f"Validation MCRMSE: {valid_mcrmse:.4f}")
        
        if valid_mcrmse < best_loss:
            best_loss = valid_mcrmse
            self.save_model(output_dir, custom_name)
            print(f"Saved model with validation MCRMSE: {best_loss:.4f}")
    
    def save_model(self, path, name, verbose = False):
        try:
            if not os.exists(path):
                os.makedirs(path)
        except:
            print("Error creating the output directory")
        
        torch.save(self.model.state_dict(), os.path.join(path, name))
        if verbose:
            print(f"Model Saved at: {os.path.join(path, name)}")