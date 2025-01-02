import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_weights = self.attention(attention_input)
        return x * attention_weights

class ImageReconstructionModel(nn.Module):
    def __init__(self):
        super(ImageReconstructionModel, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(inplace=True),
            SpatialAttention(kernel_size=7),  # Add spatial attention
            nn.Dropout(0.2)  # Reduced dropout for regularization
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128),  # Add channel attention
            nn.Dropout(0.2)  # Reduced dropout for regularization
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ChannelAttention(256),  # Add channel attention
            nn.Dropout(0.2)  # Reduced dropout for regularization
        )

        # Bottleneck with Residual Blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),  # Replace transposed convolution
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            SpatialAttention(kernel_size=7)  # Add spatial attention
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            SpatialAttention(kernel_size=7)  # Add spatial attention
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),  # Final upsample
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Skip connection layers with attention
        self.skip1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),  # Align encoder1 channels with decoder2
            nn.BatchNorm2d(128),
            ChannelAttention(128),
            ResidualBlock(128)  # Add refinement
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),  # Align encoder2 channels with bottlenecked
            nn.BatchNorm2d(256),
            ChannelAttention(256),
            ResidualBlock(256)  # Add refinement
        )

        # Refinement layer for fine details
        self.refinement = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # Adjust the input channels (48 in this case)
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),  # Additional residual block for finer details
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Final output with 3 channels
            nn.Sigmoid()  # Normalize output to [0, 1]
        )


    def forward(self, x):
        # Encode
        enc1 = self.encoder1(x)  # 64 channels
        enc2 = self.encoder2(enc1)  # 128 channels
        enc3 = self.encoder3(enc2)  # 256 channels

        # Bottleneck
        bottlenecked = self.bottleneck(enc3)

        # Decode with skip connections
        aligned_enc2 = nn.functional.interpolate(enc2, size=bottlenecked.shape[2:], mode="bilinear", align_corners=False)
        aligned_enc2 = self.skip2(aligned_enc2)  # Match channels to bottlenecked (256)

        dec1 = self.decoder1(bottlenecked + aligned_enc2)  # Combine bottleneck with encoder2 features

        aligned_enc1 = nn.functional.interpolate(enc1, size=dec1.shape[2:], mode="bilinear", align_corners=False)
        aligned_enc1 = self.skip1(aligned_enc1)  # Match channels to dec1 (128)

        dec2 = self.decoder2(dec1 + aligned_enc1)  # Combine decoder1 with encoder1 features

        # Final reconstruction
        dec3 = self.decoder3(dec2)

        # Refinement for fine details
        refined = self.refinement(dec3)

        return refined


# Load Model Function
def load_model(model_path, device="cpu"):
    model = ImageReconstructionModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()

# Image Loading Utility
def load_images(image_dir):
    images = []
    for file_name in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, file_name)
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            images.append(img)
    return images

# Run Inference
def run_inference(model, images, target_size=(224, 224), device="cpu"):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    results = []
    with torch.no_grad():
        for img in images:
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            output = model(input_tensor)
            output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_image = np.clip(output_image, 0, 1)
            results.append((output_image * 255).astype(np.uint8))
    return results
