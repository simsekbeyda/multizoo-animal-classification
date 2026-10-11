from torchvision import transforms
import torch
from src import config


def get_train_transforms():
    """Returns data augmentation and preprocessing transforms for training."""
    aug = config.AUGMENTATION_SETTINGS
    transform_list = []

    # Resize / Crop / 
    if aug.get('random_resized_crop', False):
        transform_list.append(transforms.RandomResizedCrop(config.IMAGE_SIZE))
    else:
        transform_list.append(transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    # Horizontal Flip
    flip_prob = aug.get('horizontal_flip_prob', 0)
    if flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

    # Color Jitter
    cj_args = {
        'brightness': aug.get('color_jitter_brightness', 0),
        'contrast': aug.get('color_jitter_contrast', 0),
        'saturation': aug.get('color_jitter_saturation', 0),
        'hue': aug.get('color_jitter_hue', 0)
    }
    if any(val > 0 for val in cj_args.values()):
        transform_list.append(transforms.ColorJitter(**cj_args))

    # Rotation
    rotation_deg = aug.get('random_rotation_degrees', 0)
    if rotation_deg > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_deg))

    # ToTensor & Normalize (her zaman gerekli)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

    return transforms.Compose(transform_list)


def get_val_test_transforms():
    """Returns transforms for validation/test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])




def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    '''Returns cutmix inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)

    cut_rat = torch.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # uniform
    cx = torch.randint(w, (1,)).item()
    cy = torch.randint(h, (1,)).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, w)
    bby2 = min(cy + cut_h // 2, h)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
