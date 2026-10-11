import os
import json
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from src import preprocess, config

class MultiZooDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx

        if self.class_to_idx is None and self.labels:
            sorted_classes = sorted(list(set(self._get_class_names_from_paths(image_paths))) )
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image: {img_path} | {e}")

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _get_class_names_from_paths(image_paths):
        return [os.path.basename(os.path.dirname(path)) for path in image_paths]

def get_dataloaders(data_dir, batch_size, num_workers, validation_split, random_seed):
    full_dataset = ImageFolder(root=data_dir)
    class_to_idx = full_dataset.class_to_idx

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    with open(os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE), 'w') as f:
        json.dump(class_to_idx, f)

    config.NUM_CLASSES = len(class_to_idx)
    print(f"Found {len(full_dataset.samples)} images belonging to {config.NUM_CLASSES} classes.")

    image_paths = [s[0] for s in full_dataset.samples]
    labels = [s[1] for s in full_dataset.samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=validation_split,
        random_state=random_seed,
        stratify=labels
    )

    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    train_transforms = preprocess.get_train_transforms()
    val_transforms = preprocess.get_val_test_transforms()

    train_dataset = MultiZooDataset(train_paths, train_labels, transform=train_transforms, class_to_idx=class_to_idx)
    val_dataset = MultiZooDataset(val_paths, val_labels, transform=val_transforms, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_to_idx

def get_test_dataloader(test_data_dir, batch_size, num_workers, class_to_idx):
    test_transforms = preprocess.get_val_test_transforms()

    try:
        test_dataset_imgfolder = ImageFolder(root=test_data_dir, transform=test_transforms)
        current_test_class_to_idx = test_dataset_imgfolder.class_to_idx

        image_paths = []
        labels = []

        for img_path, numeric_label in test_dataset_imgfolder.samples:
            class_name = test_dataset_imgfolder.classes[numeric_label]
            if class_name in class_to_idx:
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
            else:
                print(f"Skipping class '{class_name}' not in training set.")

        if not image_paths:
            raise FileNotFoundError("No matching classes found in test data.")

        test_dataset = MultiZooDataset(image_paths, labels, transform=test_transforms, class_to_idx=class_to_idx)
        print(f"Loaded test data using ImageFolder structure: {len(test_dataset)} images.")

    except Exception:
        print(f"Fallback: Loading test data as flat directory...")
        image_paths = glob.glob(os.path.join(test_data_dir, '*.jpg')) + \
                      glob.glob(os.path.join(test_data_dir, '*.jpeg')) + \
                      glob.glob(os.path.join(test_data_dir, '*.png'))

        if not image_paths:
            print("No image files found in test directory.")
            return None

        labels = [-1] * len(image_paths)
        test_dataset = MultiZooDataset(image_paths, labels, transform=test_transforms, class_to_idx=class_to_idx)
        print(f"Loaded test data from flat directory: {len(test_dataset)} images.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader
