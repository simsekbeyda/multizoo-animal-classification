import torch

#  Proje Dizinleri 
DATA_DIR = "data/MultiZoo/train"
TEST_DATA_DIR = "data/MultiZoo/test"
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"
BEST_MODEL_NAME = "best_multizoo_model.pth"
LEARNING_CURVE_PLOT_NAME = "learning_curve.png"
METRICS_FILE_NAME = "metrics_summary.json"
CLASS_MAPPING_FILE = "class_to_idx.json"
MODEL_PATH_FOR_APP = f"{MODEL_SAVE_DIR}/{BEST_MODEL_NAME}"

# Görüntü Boyutu & Veri
IMAGE_SIZE = 224
BATCH_SIZE = 8                   
NUM_WORKERS = 0                 
VALIDATION_SPLIT = 0.2

#  Model Ayarları 
MODEL_NAME = "swin_base_patch4_window7_224"   
PRETRAINED = True
NUM_CLASSES = None

#  Eğitim Ayarları
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "CrossEntropyLoss"

#  Early Stopping 
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

#  Normalizasyon (ImageNet) 
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

#  Data Augmentation 
AUGMENTATION_SETTINGS = {
    'random_resized_crop': True,
    'horizontal_flip_prob': 0.5,
    'color_jitter_brightness': 0.2,
    'color_jitter_contrast': 0.2,
    'color_jitter_saturation': 0.2,
    'color_jitter_hue': 0.05,
    'random_rotation_degrees': 15
}

#  Ekstra Teknikler (MixUp, CutMix, Smoothing) 
USE_MIXUP = True
USE_CUTMIX = False
LABEL_SMOOTHING = 0.1  

#  Değerlendirme
AVERAGE_METHOD = 'weighted'

# Arayüz Başlığı 
APP_TITLE = "MultiZoo Animal Classifier"

#  Rastgelelik 
RANDOM_SEED = 42
