import torch.nn as nn
from torchvision import models

def get_transfer_model():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')


    new_layer = nn.Linear(model.fc.in_features, 2)
    model.fc = new_layer

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False # Disable gradient computation
    
    return model