"""Perceptual distillation for multimodal processing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class PerceptualDistillation(nn.Module):
    """
    Perceptual distillation for transferring knowledge from large frozen encoders
    to lightweight student encoders.

    L_distill = ||f_student(x) - sg(f_teacher(x))||²

    Enables multimodal processing within 4GB GPU budget by using lightweight
    students (~30-50MB) at runtime while distilling from large teachers (~400MB+)
    during training.
    """

    def __init__(
        self,
        student_encoder: nn.Module,
        teacher_encoder: Optional[nn.Module] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher if provided
        if self.teacher_encoder is not None:
            for param in self.teacher_encoder.parameters():
                param.requires_grad = False
            self.teacher_encoder.eval()

        # Projection layer to match teacher dimension if needed
        self.projection = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute perceptual distillation loss.

        Args:
            inputs: Input data (images, audio, etc.)
            mode: "train" or "inference"

        Returns:
            Dictionary containing:
            - student_features: Student encoder output
            - distillation_loss: Distillation loss (train mode only)
        """
        # Student forward pass
        student_features = self.student_encoder(inputs)
        student_features = self.projection(student_features)

        output = {"student_features": student_features}

        # Compute distillation loss during training
        if mode == "train" and self.teacher_encoder is not None:
            with torch.no_grad():
                teacher_features = self.teacher_encoder(inputs)

            # L2 distillation loss
            distillation_loss = F.mse_loss(
                student_features,
                teacher_features.detach(),
            )

            output["distillation_loss"] = distillation_loss
            output["teacher_features"] = teacher_features

        return output

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inference using only student encoder (memory efficient)."""
        with torch.no_grad():
            student_features = self.student_encoder(inputs)
            student_features = self.projection(student_features)
        return student_features


class LightweightVisionEncoder(nn.Module):
    """Lightweight vision encoder (~30-50MB) for distillation."""

    def __init__(
        self,
        input_size: int = 224,
        hidden_size: int = 512,
        output_size: int = 512,
    ):
        super().__init__()

        # Lightweight CNN backbone
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Projection to output
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Images (batch, 3, H, W)

        Returns:
            features: Encoded features (batch, output_size)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        return features


class LightweightAudioEncoder(nn.Module):
    """Lightweight audio encoder (~30-50MB) for distillation."""

    def __init__(
        self,
        input_size: int = 80,  # Mel spectrogram bins
        hidden_size: int = 512,
        output_size: int = 512,
    ):
        super().__init__()

        # Lightweight CNN for audio
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, hidden_size, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Mel spectrograms (batch, 1, freq, time)

        Returns:
            features: Encoded features (batch, output_size)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        return features
