import cv2
import numpy as np

def apply_shadow_eraser(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in the LAB color space
    to neutralize shadows and equalize lighting across the road surface.
    Expects RGB image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def get_texture_map(image):
    """
    Enhances road texture by extracting roughness patterns.
    Uses grayscale conversion, contrast enhancement, and Sobel edge detection.
    Returns a 3-channel image blended with the original, compatible with ResNet-18.
    Expects RGB image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast before edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Sobel edge detection for roughness/texture
    sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude of edges
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Avoid division by zero
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = np.uint8(255 * magnitude / max_val)
    else:
        magnitude = np.uint8(magnitude)
    
    texture_3ch = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    
    # Blending shadow-neutralized original image with texture map to create a texture-enhanced output
    enhanced = cv2.addWeighted(image, 0.7, texture_3ch, 0.3, 0)
    return enhanced
