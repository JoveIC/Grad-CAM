
import cv2
import torch
import numpy as np

def fuse_heatmap_image(img, heatmap, resize=None, keep_heatmap=False):
    img = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)
    heatmap = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else heatmap

    if not resize:
        size = img.shape
    else:
        size = resize
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.float32(cv2.resize(heatmap, size))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    fused = np.float32(cv2.resize(img/255, size)) + np.float32(heatmap/255, size)
    fused = np.uint8((fused / np.max(fused)) * 255)
    if keep_heatmap:
        return fused, heatmap
    else:
        return heatmap

