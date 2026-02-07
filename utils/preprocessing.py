import cv2
import numpy as np
from torchvision import transforms

# CT windowing

def ct_windowing(img, center=-600,width=1500):
    img = img.astype(np.float32)

    lower = center - width // 2
    upper = center + width // 2

    img = np.clip(img,lower,upper)
    img = (img - lower) / (upper - lower + 1e8)
    img = (img * 255).astype(np.uint8)

# Individual preprocessing operations
def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced,cv2.COLOR_GRAY2RGB)

def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)

def gaussian_denoise(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def median_denoise(img):
    return cv2.medianBlur(img, 5)

def sharpen(img):
    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])
    return cv2.filter2D(img,-1,kernel)

def normalize_minmax(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


def normalize_zscore(img):
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img) + 1e-8
    img = (img - mean) / std
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    return img


class CTPreprocess:
    def __init__(self,
        windowing=False,
        clahe=False,
        hist_eq=False,
        gaussian=False,
        median=False,
        sharpen_flag=False,
        norm_type="minmax"):
        self.windowing = windowing
        self.clahe = clahe
        self.hist_eq = hist_eq
        self.gaussian = gaussian
        self.median = median
        self.sharpen_flag = sharpen_flag
        self.norm_type = norm_type

    def __call__(self, image):
        img = np.array(image)

        if self.windowing:
            img = ct_windowing(img)

        if self.clahe:
            img = apply_clahe(img)
        elif self.hist_eq:
            img = histogram_equalization(img)

        if self.gaussian:
            img = gaussian_denoise(img)
        if self.median:
            img = median_denoise(img)

        if self.sharpen_flag:
            img = sharpen(img)

        if self.norm_type == "minmax":
            img = normalize_minmax(img)
        elif self.norm_type == "zscore":
            img = normalize_zscore(img)

        return img
    
def get_transformss(
    img_size=224,
    train=True,
    **preprocess_kwargs
):
    preprocess = CTPreprocess(**preprocess_kwargs)

    if train:
        return transforms.Compose([
            preprocess,
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),

            # Augmenation
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.1)),

            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        return transforms.Compose([
            preprocess,
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])