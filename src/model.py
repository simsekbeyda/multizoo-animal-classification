import timm
import torch.nn as nn
from src import config

def get_model(num_classes, pretrained=True, model_name='swin_base_patch4_window7_224'):
    """
    Loads a pre-trained Vision Transformer model and adapts its classifier head
    for the required number of classes.
    """
   
    model = timm.create_model(model_name, pretrained=pretrained)

   
    
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear): 
        in_features = model.head.in_features
        
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear): 
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): 
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        
        try:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            print(f"Re-created model {model_name} with num_classes={num_classes}.")
        except Exception as e:
            raise AttributeError(f"Could not find a standard classifier head (head, fc, classifier) for {model_name} and failed to re-create with num_classes. You might need to inspect the model structure and adapt this function. Error: {e}")

    print(f"Loaded model: {model_name}")
    if pretrained:
        print("Using pretrained weights.")
    print(f"Adapted classifier for {num_classes} classes.")
    
    return model

if __name__ == '__main__':
    
    config.NUM_CLASSES = 10 
    if config.NUM_CLASSES is None:
        print("Error: config.NUM_CLASSES is not set. Please set it before creating the model.")
    else:
        print(f"Attempting to create model for {config.NUM_CLASSES} classes.")
        try:
            test_model = get_model(num_classes=config.NUM_CLASSES, pretrained=True, model_name=config.MODEL_NAME)
            print("Model created successfully.")
            
            
            
            import torch
            dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            output = test_model(dummy_input)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error during model creation or test: {e}") 