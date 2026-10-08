"""Gradient magnitude used for feature-patch verification and NCC extension.

Copied unchanged from propose_fixed_features.py; the earlier proposal UI is omitted.
"""
import cv2
import numpy as np

def gradient_magnitude(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)
