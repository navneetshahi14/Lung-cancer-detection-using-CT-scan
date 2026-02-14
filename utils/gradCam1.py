import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()

        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        loss = output[:, class_idx]

        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grads = self.gradients[0]        # [C,H,W]
        acts = self.activations[0]       # [C,H,W]

        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(1, 2))  # [C]

        # Weighted combination
        cam = torch.sum(weights[:, None, None] * acts, dim=0)

        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.cpu().numpy(), class_idx

def overlay_cam_on_image(img_np, cam, alpha=0.4):
    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay


def show_gradcam(img, overlay, pred_class, class_names):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM → {class_names[pred_class]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
