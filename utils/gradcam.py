"""
gradcam.py — Grad-CAM implementation for DeepGuard / Sach-AI.

Generates heatmaps to localize manipulated facial regions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ImageModel (DenseNet121).
    """

    def __init__(self, model, target_layer_name="backbone.features"):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]
        self.gradients = None
        self.activations = None

    def generate(self, input_tensor):
        """
        Generate a Grad-CAM heatmap for the given input tensor.
        """
        self.model.eval()
        
        # ── Capture Activations & Gradients ───────────────────────────────────
        def save_gradient(grad):
            self.gradients = grad

        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(save_gradient)

        hook = self.target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        output = self.model(input_tensor)
        hook.remove() # Clean up the hook immediately
        
        # Target the probability of 'fake' (binary classification)
        # If output is (B, 1), we can just use that
        target = output
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target.sum().backward(retain_graph=True)

        # Get weighted combination of feature maps
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam


def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """
    Overlay a Grad-CAM heatmap on an image.
    """
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Ensure img_np is RGB and resized to 224
    img_res = cv2.resize(img_np, (224, 224))
    
    overlayed = cv2.addWeighted(img_res, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed
