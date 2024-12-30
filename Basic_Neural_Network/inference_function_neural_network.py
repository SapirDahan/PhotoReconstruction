import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from find_black_rectangle import find_black_rectangle

class ColorPredictor(nn.Module):
    def __init__(self, grid_size):
        super(ColorPredictor, self).__init__()
        self.grid_size = grid_size

        # Convolutional and pooling layers (deeper with more filters)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224 -> 112x112

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 112x112 -> 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 112x112 -> 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112 -> 56x56

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 56x56 -> 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 56x56 -> 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 56x56 -> 28x28
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 4096),  # Output size of the final conv block
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),  # Balanced regularization
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, grid_size**2 * 3),  # Output: grid_size^2 regions × 3 RGB values
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def run_inference_neural_network(model_path, images, grid_size=4, target_size=(224, 224), device="cpu"):
    """
    Perform inference on a list of images and return images with predictions overlaid.
    Args:
        model_path (str): Path to the saved .pth model file.
        images (list): List of PIL.Image objects or NumPy arrays.
        grid_size (int): Grid size (e.g., 4 for a 4x4 grid).
        target_size (tuple): Target size for image preprocessing.
        device (str): Device to use ('cpu' or 'cuda').
    Returns:
        list: A list of images with overlaid predictions.
    """
    # Load the model
    model = ColorPredictor(grid_size=grid_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    overlaid_images = []
    with torch.no_grad():
        for image in images:
            # Ensure the input is a PIL.Image
            if isinstance(image, np.ndarray):
                if image.max() <= 1:  # If normalized, scale to 0-255
                    image = (image * 255).astype(np.float32)
                image = Image.fromarray(image)

            # Preprocess and run inference
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            grid_prediction = output.view(grid_size, grid_size, 3).cpu().numpy()

            # Overlay predictions on the image
            image_np = np.array(image)  # Convert PIL.Image to NumPy array
            overlay_image = overlay_predictions_on_black_rectangle(image_np, grid_prediction, grid_size)
            overlaid_images.append(overlay_image)

    return overlaid_images

# Function: Overlay predictions onto black rectangular region
def overlay_predictions_on_black_rectangle(masked_image, predictions, grid_size):
    """Overlay the predicted RGB grid on the black rectangle."""
    # Convert to NumPy array if needed
    if isinstance(masked_image, Image.Image):
        masked_image = np.array(masked_image)
    elif isinstance(masked_image, torch.Tensor):
        masked_image = masked_image.permute(1, 2, 0).cpu().numpy()

    masked_image = masked_image.copy()  # Avoid modifying the original image
    masked_image = (masked_image * 255).astype(np.uint8) if masked_image.max() <= 1 else masked_image

    # Find the black rectangle
    y_min, y_max, x_min, x_max = find_black_rectangle(masked_image / 255.0)

    # Calculate grid cell dimensions
    region_height = (y_max - y_min) // grid_size
    region_width = (x_max - x_min) // grid_size

    # Ensure predictions is reshaped correctly: grid_size x grid_size x 3
    predictions = predictions.reshape(grid_size, grid_size, 3)

    # Overlay predicted colors on the grid
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = y_min + i * region_height
            y2 = min(y1 + region_height, y_max)
            x1 = x_min + j * region_width
            x2 = min(x1 + region_width, x_max)

            # Retrieve the predicted RGB color and apply to the region
            r, g, b = (predictions[i, j] * 255).astype(int)
            masked_image[y1:y2, x1:x2] = [r, g, b]

    return masked_image