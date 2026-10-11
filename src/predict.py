import torch
from PIL import Image
import os
import json

from src import config, utils, preprocess, model as model_loader


def predict_image(image_path, model_path=None, class_mapping_path=None):
    if model_path is None:
        model_path = config.MODEL_PATH_FOR_APP
    if class_mapping_path is None:
        class_mapping_path = os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at {class_mapping_path}")

   
    class_to_idx, idx_to_class = utils.load_class_mapping(class_mapping_path)
    num_classes = len(idx_to_class)

    
    model, _, _, _ = utils.load_model(model_path, num_classes_override=num_classes)
    model.eval().to(config.DEVICE)

    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    transform = preprocess.get_val_test_transforms()
    img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = idx_to_class.get(predicted_idx.item(), "Unknown")
    return predicted_class_name, confidence.item()


if __name__ == '__main__':
    
    sample_image = "sample_test_image.png"
    if not os.path.exists(sample_image):
        img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')
        img.save(sample_image)

    try:
        label, score = predict_image(sample_image)
        print(f"Prediction: {label}, Confidence: {score:.4f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
