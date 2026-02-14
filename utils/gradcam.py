import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()

        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )


        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx


def overlay_cam_on_image(img, cam):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay


def save_gradcam_samples(
    gradcam,
    images,
    class_names,
    save_dir,
    device,
    max_images=5
):
    """
    Generates and saves Grad-CAM overlays for sample images.
    """

    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(len(images), max_images)):

        img_tensor = images[i].unsqueeze(0).to(device)

        cam, pred_class = gradcam.generate(img_tensor)

        # tensor → numpy RGB
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = np.uint8(255 * img_np)

        overlay = overlay_cam_on_image(img_np, cam)

        save_path = os.path.join(
            save_dir,
            f"gradcam_{i}_{class_names[pred_class]}.png"
        )

        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"✅ Grad-CAM samples saved → {save_dir}")


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
