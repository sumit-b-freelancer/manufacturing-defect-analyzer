import torch
import torch.nn as nn
import timm

class SentinelModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', num_classes=2, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone model (e.g., 'tf_efficientnetv2_s', 'vit_base_patch16_224').
            num_classes (int): Number of classes for classification.
            pretrained (bool): Whether to use pretrained weights.
        """
        super(SentinelModel, self).__init__()
        
        # Load backbone from timm
        # num_classes=0 removes the classifier head, global_pool='' keeps the feature map if needed, 
        # but usually for classification we want the pooled output or we let timm handle the head.
        # Here we let timm create the backbone and replace the head or use num_classes directly if standard.
        # However, for fine-grained control and explainability (Grad-CAM), accessing the last conv layer is useful.
        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Save model specifics for Grad-CAM
        self.model_name = model_name

    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        """
        Returns the last convolutional layer for Grad-CAM.
        This depends on the architecture.
        """
        if 'efficientnet' in self.model_name:
            # EfficientNet usually has .conv_head or .blocks[-1]
            # specific implementation might vary in timm version, but often:
            return self.model.conv_head
        elif 'resnet' in self.model_name:
            return self.model.layer4
        elif 'vit' in self.model_name:
            # ViT is transformer based, Grad-CAM is different (Attention Rollout), 
            # but for LayerCAM we might look at the last block.
            # Returning None or handling separately for ViT.
            return self.model.blocks[-1].norm1 # Example, needs specific adjustment for ViT logic
        else:
            return None

if __name__ == "__main__":
    # Quick test
    model = SentinelModel(model_name='tf_efficientnetv2_s', num_classes=5)
    print(f"Model {model.model_name} created.")
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
