import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

class ExplainableAI:
    def __init__(self, model):
        self.model = model
        self.target_layers = [model.get_last_conv_layer()]
        self.cam = GradCAM(model=model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor, original_image_np=None, target_class=None):
        """
        Args:
            input_tensor: (1, C, H, W) torch tensor
            original_image_np: (H, W, 3) float32 numpy array [0, 1] for visualization overlay
            target_class: int, class index to visualize. If None, highest scoring class is used.
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        # Generate grayscale cam
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        # grayscale_cam is (1, H, W)
        grayscale_cam = grayscale_cam[0, :]

        visualization = None
        if original_image_np is not None:
            visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
        
        return grayscale_cam, visualization
