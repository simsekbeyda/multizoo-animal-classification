import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from src import config, dataset, model as model_loader, utils
from src.utils import mixup_data, cutmix_data  

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        log_probs = self.log_softmax(x)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=True, path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model_state = None
        self.best_epoch = 0
        self.best_optimizer_state = None
        self.path = path or os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, val_loss, model, epoch, optimizer):
        score = -val_loss

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_model_state = model.state_dict()
            self.best_optimizer_state = optimizer.state_dict()
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved. Saving model at epoch {epoch+1}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_best_model(self):
        if self.best_model_state:
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.best_optimizer_state,
                'loss': self.val_loss_min,
                'num_classes': config.NUM_CLASSES,
                'model_name': config.MODEL_NAME,
                'image_size': config.IMAGE_SIZE
            }, self.path)
            print(f"Best model saved to {self.path} (epoch {self.best_epoch+1})")
        else:
            print("No improvement, model not saved.")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    use_mixup = config.USE_MIXUP
    use_cutmix = config.USE_CUTMIX

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return running_loss / total, correct / total, precision, recall, f1

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return running_loss / total, correct / total, precision, recall, f1

def run_training():
    utils.set_seed(config.RANDOM_SEED)
    torch.cuda.empty_cache()

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")

    train_loader, val_loader, class_to_idx = dataset.get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        validation_split=config.VALIDATION_SPLIT,
        random_seed=config.RANDOM_SEED
    )

    config.NUM_CLASSES = len(class_to_idx)

    model = model_loader.get_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    ).to(config.DEVICE)

    if config.LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingLoss(classes=config.NUM_CLASSES, smoothing=config.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    early_stopper = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        path=os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_prec': [], 'val_prec': [], 'train_rec': [], 'val_rec': [], 'train_f1': [], 'val_f1': []}
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(model, val_loader, criterion, config.DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_prec'].append(train_prec)
        history['val_prec'].append(val_prec)
        history['train_rec'].append(train_rec)
        history['val_rec'].append(val_rec)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        early_stopper(val_loss, model, epoch, optimizer)
        if early_stopper.early_stop:
            print("Early stopping activated.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time//60:.0f}m {total_time%60:.0f}s")

    early_stopper.save_best_model()

    utils.plot_learning_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        os.path.join(config.RESULTS_DIR, config.LEARNING_CURVE_PLOT_NAME)
    )

    utils.plot_precision_recall_f1(
        history['train_prec'], history['val_prec'],
        history['train_rec'], history['val_rec'],
        history['train_f1'], history['val_f1'],
        os.path.join(config.RESULTS_DIR, "prf_curve.png")
    )

if __name__ == "__main__":
    run_training()