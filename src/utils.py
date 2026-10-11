import torch
import os
import matplotlib.pyplot as plt
import json
from src import config
import random
import numpy as np

def save_model(model, model_path, epoch, optimizer, loss):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'num_classes': config.NUM_CLASSES,
        'model_name': config.MODEL_NAME,
        'image_size': config.IMAGE_SIZE
    }, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path, num_classes_override=None):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None, None, None

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    num_classes_to_load = num_classes_override if num_classes_override is not None else checkpoint.get('num_classes')
    if num_classes_to_load is None:
        class_mapping_path = os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                class_to_idx = json.load(f)
            num_classes_to_load = len(class_to_idx)
            print(f"Loaded num_classes ({num_classes_to_load}) from {config.CLASS_MAPPING_FILE} as it was not in checkpoint.")
        else:
            raise ValueError("Number of classes not found in checkpoint or class_to_idx.json. Cannot load model.")

    timm_model_name = checkpoint.get('model_name', config.MODEL_NAME)
    from src.model import get_model
    model = get_model(num_classes=num_classes_to_load, pretrained=False, model_name=timm_model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)

    epoch = checkpoint.get('epoch', 0)
    optimizer_state_dict = checkpoint.get('optimizer_state_dict')
    loss = checkpoint.get('loss')
    print(f"Model loaded from {model_path}. Trained for {epoch} epochs. Last loss: {loss:.4f}")
    return model, epoch, optimizer_state_dict, loss

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Learning curves saved to {save_path}")
    plt.close()

def plot_precision_recall_f1(train_prec, val_prec, train_rec, val_rec, train_f1, val_f1, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    epochs = range(1, len(train_prec) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_prec, label='Train Precision', linestyle='--', marker='o')
    plt.plot(epochs, val_prec, label='Val Precision', linestyle='-', marker='o')
    plt.plot(epochs, train_rec, label='Train Recall', linestyle='--', marker='x')
    plt.plot(epochs, val_rec, label='Val Recall', linestyle='-', marker='x')
    plt.plot(epochs, train_f1, label='Train F1-Score', linestyle='--', marker='s')
    plt.plot(epochs, val_f1, label='Val F1-Score', linestyle='-', marker='s')

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1-score over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"PRF curves saved to {save_path}")
    plt.close()

def save_metrics(metrics, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")

def load_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Metrics file not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    print(f"Metrics loaded from {file_path}")
    return metrics

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

def load_class_mapping(file_path=os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)):
    if not os.path.exists(file_path):
        print(f"Class mapping file not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class

def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    rx = np.random.randint(w)
    ry = np.random.randint(h)
    rw = int(w * np.sqrt(1 - lam))
    rh = int(h * np.sqrt(1 - lam))
    x1 = np.clip(rx - rw // 2, 0, w)
    x2 = np.clip(rx + rw // 2, 0, w)
    y1 = np.clip(ry - rh // 2, 0, h)
    y2 = np.clip(ry + rh // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam