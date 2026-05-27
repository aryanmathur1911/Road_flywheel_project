import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from .data_file import apply_shadow_eraser, get_texture_map

class TextureVisionAnalyzer:
    def __init__(self, model_path="models/road_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        # 2. Hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        self.model.layer4.register_forward_hook(self.save_act)
        self.model.layer4.register_full_backward_hook(self.save_grad)

    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]

    def visualize(self, image_path, output_path="test_step_2.jpg"):
        # 1️⃣ Load original image
        img_cv = cv2.imread(image_path)
        if img_cv is None: return
        h, w = img_cv.shape[:2]

        # 2️⃣ Pre-process (Shadow Eraser + Texture)
        clean_img = apply_shadow_eraser(img_cv)
        texture_img = get_texture_map(clean_img)
        
        # 3️⃣ Model Inference
        # Convert to Tensor for PyTorch
        pil_img = Image.fromarray(texture_img)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)

        # 4️⃣ Forward Pass & Backward for Grad-CAM
        output = self.model(input_tensor)
        score = output[:, 0]
        self.model.zero_grad()
        score.backward()

        # 5️⃣ --- THE "BYPASS" RESIZE LOGIC ---
        # Instead of cv2.resize, we use PyTorch Interpolate
        # self.activations is [1, 512, 7, 7]
        heatmap_tensor = torch.mean(self.activations, dim=1, keepdim=True) # [1, 1, 7, 7]
        heatmap_tensor = F.relu(heatmap_tensor)
        
        # Resize from 7x7 to Original Image size (h, w)
        # This uses PyTorch's engine, completely skipping the OpenCV resize bug
        heatmap_resized = F.interpolate(heatmap_tensor, size=(h, w), mode='bilinear', align_corners=False)
        heatmap_np = heatmap_resized.squeeze().detach().cpu().numpy()
        
        # Normalize 0-1
        denom = heatmap_np.max() - heatmap_np.min()
        cam_norm = (heatmap_np - heatmap_np.min()) / (denom + 1e-8)

        # 6️⃣ Final Blending (Using CV2 only for the color map)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        result = cv2.addWeighted(img_cv, 0.7, heatmap_color, 0.3, 0)

        # Labeling
        label = "ACTUAL HAZARD" if score.item() > 0 else "SAFE ROAD"
        cv2.putText(result, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        cv2.imwrite(output_path, result)
        print(f"✅ Success! Generated {output_path} without OpenCV resize.")

if __name__ == "__main__":
    analyzer = TextureVisionAnalyzer()
    analyzer.visualize("test_step_1.jpg")